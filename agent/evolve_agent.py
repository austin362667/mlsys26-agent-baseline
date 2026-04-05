# Evolve Agent: propose kernels in a pool, evolve with elite selection

import argparse
import json
import logging
import os
import shutil

import numpy as np
from tqdm import tqdm

from agent.eval import EvalResult, calculate_score
from agent.iterative_agent import propose_step, run_iterative_loop

logger = logging.getLogger(__name__)


def load_evolve_from_logs(log_path: str):
    """Load completed evolve proposals from an existing log directory."""
    empty = ([], [], [], [], [], [], 0)

    if not os.path.exists(log_path):
        logger.warning(f"Resume path {log_path} does not exist")
        return empty

    step_files: list[tuple[int, str]] = []
    for filename in os.listdir(log_path):
        if not (filename.startswith("proposal_") and filename.endswith(".py")):
            continue
        suffix = filename[len("proposal_") : -3]
        if not suffix.isdigit():
            continue
        step_files.append((int(suffix), filename))

    if not step_files:
        logger.warning(f"No evolve proposal files found in {log_path}")
        return empty

    step_files.sort(key=lambda x: x[0])
    kernel_pool: list[str] = []
    metrics_pool: list[EvalResult] = []
    proposal_ids: list[int] = []
    max_step = 0

    for step_idx, kernel_file in step_files:
        kernel_path = os.path.join(log_path, kernel_file)
        metrics_path = os.path.join(log_path, f"proposal_{step_idx}_metrics.json")
        if not os.path.exists(metrics_path):
            logger.warning(
                f"Metrics file {metrics_path} not found, skipping step {step_idx}"
            )
            continue

        with open(kernel_path, "r") as f:
            kernel = f.read()
        with open(metrics_path, "r") as f:
            metrics = EvalResult(**json.load(f))

        kernel_pool.append(kernel)
        metrics_pool.append(metrics)
        proposal_ids.append(step_idx)
        max_step = max(max_step, step_idx)

    if not proposal_ids:
        logger.warning(f"No complete evolve steps found in {log_path}")
        return empty

    sorted_data = sorted(
        zip(kernel_pool, metrics_pool, proposal_ids),
        key=lambda x: calculate_score(x[1]),
        reverse=True,
    )
    elite_pool, elite_metrics_pool, elite_proposal_ids = zip(*sorted_data)
    logger.info(
        f"Resumed evolve state from {log_path}: loaded {len(proposal_ids)} proposals, "
        f"max_step={max_step}, best_score={calculate_score(elite_metrics_pool[0])}"
    )
    return (
        kernel_pool,
        metrics_pool,
        proposal_ids,
        list(elite_pool),
        list(elite_metrics_pool),
        list(elite_proposal_ids),
        max_step,
    )


def copy_evolve_step_files(src_path: str, dst_path: str, max_step: int = 0):
    """Copy evolve proposal logs into a new output directory when resuming elsewhere."""
    os.makedirs(dst_path, exist_ok=True)

    for step_idx in range(1, max_step + 1):
        for suffix in (".py", ".txt", "_metrics.json", "_log.json"):
            src_file = os.path.join(src_path, f"proposal_{step_idx}{suffix}")
            if os.path.exists(src_file):
                dst_file = os.path.join(dst_path, f"proposal_{step_idx}{suffix}")
                shutil.copy2(src_file, dst_file)


def run_evolve_loop(
    ref_arch_src: str,
    inference_server: str,
    args: argparse.Namespace,
    log_path: str = None,
):
    """Run the evolve loop: propose kernels, maintain elite pool, optionally refine."""
    start_step = 0
    kernel_pool: list[str] = []
    metrics_pool: list[EvalResult] = []
    proposal_ids: list[int] = []
    elite_pool: list[str] = []
    elite_metrics_pool: list[EvalResult] = []
    elite_proposal_ids: list[int] = []

    if args.agent_type == "evolve" and args.resume_from is not None:
        (
            kernel_pool,
            metrics_pool,
            proposal_ids,
            elite_pool,
            elite_metrics_pool,
            elite_proposal_ids,
            start_step,
        ) = load_evolve_from_logs(args.resume_from)
        if start_step > 0 and log_path and log_path != args.resume_from:
            copy_evolve_step_files(args.resume_from, log_path, start_step)
        if start_step >= args.proposal_steps and elite_pool:
            logger.info(
                f"Resume already reached target total_steps={args.proposal_steps}, "
                "returning cached best kernel"
            )
            return elite_pool[0], elite_metrics_pool[0]

    def _last_k_idx(n: int, k: int) -> list[int]:
        if k <= 0 or n <= 0:
            return []
        return list(range(max(0, n - k), n))

    for i in tqdm(
        range(start_step, args.proposal_steps),
        desc=f"Evolve Agent on {args.level}_{args.problem_id}",
        initial=start_step,
        total=args.proposal_steps,
    ):
        logger.debug(f"Running proposal {i+1} of {args.proposal_steps}")

        recent_context_ids: list[int] = []
        elite_context_ids: list[int] = []
        recent_idx: list[int] = []
        elite_idx: list[int] = []
        pool_size = int(getattr(args, "pool_size", 0) or 0)

        if args.agent_type == "evolve":
            k_recent = pool_size // 2
            k_elite = pool_size - k_recent

            recent_idx = _last_k_idx(len(proposal_ids), k_recent)
            recent_context_ids = [proposal_ids[j] for j in recent_idx]

            # Sample elite (correct-only) by softmax over speedup
            if k_elite > 0 and elite_metrics_pool:
                candidate_idx: list[int] = []
                candidate_speedups: list[float] = []
                for j, m in enumerate(elite_metrics_pool):
                    score = calculate_score(m)
                    if (
                        score[0] == 1
                        and score[1] == 1
                        and elite_proposal_ids[j] not in recent_context_ids
                    ):
                        candidate_idx.append(j)
                        candidate_speedups.append(score[2])

                if len(candidate_idx) > 0:
                    choose_n = min(k_elite, len(candidate_idx))
                    tau = float(getattr(args, "softmax_temperature", 1.0))
                    tau = max(1e-6, tau)
                    x = np.asarray(candidate_speedups, dtype=np.float64) / tau
                    x = x - np.max(x)
                    p = np.exp(x)
                    p = p / np.sum(p)
                    picked_local = np.random.choice(
                        len(candidate_idx), size=choose_n, replace=False, p=p
                    )
                    picked_idx = [candidate_idx[t] for t in picked_local.tolist()]
                    elite_context_ids = [elite_proposal_ids[j] for j in picked_idx]
                    elite_idx = picked_idx
        else:
            raise ValueError(f"Invalid agent type for evolve loop: {args.agent_type}")

        recent_kernels_to_pass = (
            [kernel_pool[j] for j in recent_idx] if recent_idx else []
        )
        recent_metrics_to_pass = (
            [metrics_pool[j] for j in recent_idx] if recent_idx else []
        )
        elite_kernels_to_pass = [elite_pool[j] for j in elite_idx] if elite_idx else []
        elite_metrics_to_pass = (
            [elite_metrics_pool[j] for j in elite_idx] if elite_idx else []
        )

        proposal_kernel, proposal_metrics, logs = propose_step(
            ref_arch_src,
            inference_server,
            recent_kernels_to_pass,
            recent_metrics_to_pass,
            args,
            context_ids=recent_context_ids,
            elite_kernel_pool=elite_kernels_to_pass,
            elite_metrics_pool=elite_metrics_to_pass,
            elite_context_ids=elite_context_ids,
        )

        # Log the proposal
        if log_path is not None:
            with open(os.path.join(log_path, f"proposal_{i+1}.txt"), "w") as f:
                f.write(logs["proposer_prompt"])
            with open(os.path.join(log_path, f"proposal_{i+1}.py"), "w") as f:
                f.write(logs["proposal_kernel"])
            with open(os.path.join(log_path, f"proposal_{i+1}_metrics.json"), "w") as f:
                json.dump(logs["proposal_metrics"].model_dump(), f)

            step_log = {
                "step": i + 1,
                "step_type": "large_step",
                "context": {
                    "recent": recent_context_ids,
                    "elite": elite_context_ids,
                },
                "context_metrics": {
                    "recent": [m.model_dump() for m in recent_metrics_to_pass],
                    "elite": [m.model_dump() for m in elite_metrics_to_pass],
                },
                "context_size": {
                    "recent": len(recent_context_ids),
                    "elite": len(elite_context_ids),
                },
                "compiled": proposal_metrics.compiled if proposal_metrics else False,
                "correct": (
                    proposal_metrics.correct if proposal_metrics else False
                ),
                "speedup": proposal_metrics.speedup if proposal_metrics else 0.0,
                "speedup_per_workload": (
                    proposal_metrics.speedup_per_workload if proposal_metrics else []
                ),
                "score": calculate_score(proposal_metrics),
            }
            with open(os.path.join(log_path, f"proposal_{i+1}_log.json"), "w") as f:
                json.dump(step_log, f, indent=2)

        if args.refine_steps > 0:
            local_best_kernel, local_best_metrics = run_iterative_loop(
                ref_arch_src,
                inference_server,
                proposal_kernel,
                proposal_metrics,
                args,
                large_loop_id=i + 1,
                log_path=log_path,
            )
        else:
            local_best_kernel = proposal_kernel
            local_best_metrics = proposal_metrics

        logger.debug(f"Local Best Metrics: {local_best_metrics}")
        logger.debug(f"Local Best Score: {calculate_score(local_best_metrics)}")
        kernel_pool.append(local_best_kernel)
        metrics_pool.append(local_best_metrics)
        proposal_ids.append(i + 1)

        # Maintain a separately sorted elite pool for best-kernel context
        sorted_data = sorted(
            zip(kernel_pool, metrics_pool, proposal_ids),
            key=lambda x: calculate_score(x[1]),
            reverse=True,
        )
        elite_pool, elite_metrics_pool, elite_proposal_ids = zip(*sorted_data)
        elite_pool = list(elite_pool)
        elite_metrics_pool = list(elite_metrics_pool)
        elite_proposal_ids = list(elite_proposal_ids)

    if len(elite_pool) == 0:
        raise ValueError("No kernels were generated; empty pool.")

    return elite_pool[0], elite_metrics_pool[0]
