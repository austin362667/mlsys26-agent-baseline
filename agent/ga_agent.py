import argparse
import json
import logging
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from agent.eval import EvalResult, calculate_score
from agent.iterative_agent import propose_step
from prompt.ga_prompt import generate_crossover_prompt

logger = logging.getLogger(__name__)

MAX_WORKERS = 3

def _fitness(metric: EvalResult | None, worst_correct_speedup: float) -> float:
    """
    Map an EvalResult to a single positive scalar for softmax selection.

    - Correct kernel      : geometric mean speedup  (>= 0)
    - Compile-but-wrong   : worst_correct_speedup / 10  (very low but nonzero)
    - Compile failure/None: worst_correct_speedup / 100 (even lower)

    When no correct kernel exists yet, worst_correct_speedup defaults to 1.0
    so penalties are still meaningful relative to each other.
    """
    base = max(worst_correct_speedup, 1e-6)
    if metric is None or not metric.compiled:
        return base / 100.0
    if not metric.correct:
        return base / 10.0
    score = calculate_score(metric)
    return max(score[2], 1e-9)


def _selection_probs(
    metrics: list[EvalResult | None],
    tau: float = 1.0,
) -> np.ndarray:
    """Softmax over fitness scalars with temperature tau."""
    correct_speedups = [
        calculate_score(m)[2]
        for m in metrics
        if m is not None and m.compiled and m.correct
    ]
    worst_correct = min(correct_speedups) if correct_speedups else 1.0

    fitnesses = np.array(
        [_fitness(m, worst_correct) for m in metrics], dtype=np.float64
    )
    x = fitnesses / max(tau, 1e-6)
    x -= x.max()
    p = np.exp(x)
    return p / p.sum()


def _sample_two_parents(
    pool: list[str],
    metrics: list[EvalResult | None],
    tau: float,
) -> tuple[int, int]:
    """Sample two distinct parent indices without replacement."""
    if len(pool) < 2:
        return 0, 0
        
    probs = _selection_probs(metrics, tau)
    idx_a = int(np.random.choice(len(pool), p=probs))
    # Zero out idx_a and renormalize for second draw
    probs[idx_a] = 0.0
    total = probs.sum()
    if total < 1e-12:
        # All mass was on idx_a
        probs = np.ones(len(pool), dtype=np.float64)
        probs[idx_a] = 0.0
        total = probs.sum()
    probs /= total
    idx_b = int(np.random.choice(len(pool), p=probs))
    return idx_a, idx_b


def _load_ga_from_logs(log_path: str, population_size: int):
    """Load a GA population from an existing log directory."""
    empty = ([], [], [], 0)
    if not os.path.exists(log_path):
        logger.warning(f"Resume path {log_path} does not exist")
        return empty

    entries: list[tuple[int, str, EvalResult]] = []
    for filename in os.listdir(log_path):
        if not (filename.startswith("kernel_") and filename.endswith(".py")):
            continue
        suffix = filename[len("kernel_"):-3]
        if not suffix.isdigit():
            continue
        kid = int(suffix)
        metrics_path = os.path.join(log_path, f"kernel_{kid}_metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(os.path.join(log_path, filename)) as f:
            code = f.read()
        with open(metrics_path) as f:
            metrics = EvalResult(**json.load(f))
        entries.append((kid, code, metrics))

    if not entries:
        logger.warning(f"No complete GA kernel files found in {log_path}")
        return empty

    # Sort by fitness descending, keep top population_size
    entries.sort(key=lambda e: calculate_score(e[2]), reverse=True)
    entries = entries[:population_size]

    max_kid = max(e[0] for e in entries)
    pool    = [e[1] for e in entries]
    metrics = [e[2] for e in entries]
    ids     = [e[0] for e in entries]

    logger.info(
        f"Resumed GA from {log_path}: {len(pool)} individuals, "
        f"max_id={max_kid}, best={calculate_score(metrics[0])}"
    )
    return pool, metrics, ids, max_kid


def _save_kernel(log_path: str, kid: int, code: str, metrics: EvalResult):
    if log_path is None:
        return
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, f"kernel_{kid}.py"), "w") as f:
        f.write(code)
    with open(os.path.join(log_path, f"kernel_{kid}_metrics.json"), "w") as f:
        json.dump(metrics.model_dump(), f, indent=2)


def _save_generation_log(
    log_path: str,
    generation: int,
    offspring_ids: list[int],
    pool_ids: list[int],
    pool_metrics: list[EvalResult],
):
    if log_path is None:
        return
    log = {
        "generation": generation,
        "offspring_ids": offspring_ids,
        "surviving_pool_ids": pool_ids,
        "best_score": calculate_score(pool_metrics[0]) if pool_metrics else None,
        "pool_scores": [calculate_score(m) for m in pool_metrics],
    }
    with open(os.path.join(log_path, f"generation_{generation}_log.json"), "w") as f:
        json.dump(log, f, indent=2)


def _generate_offspring(
    ref_arch_src: str,
    inference_server: str,
    pool: list[str],
    metrics: list[EvalResult | None],
    args: argparse.Namespace,
    kid: int,
    log_path: str | None,
) -> tuple[str, EvalResult, int]:
    """Sample two parents, crossover via LLM, eval, return (code, metrics, kid)."""
    tau = float(getattr(args, "softmax_temperature", 1.0))
    idx_a, idx_b = _sample_two_parents(pool, metrics, tau)

    # Ensure parent_a is always the better one
    score_a = calculate_score(metrics[idx_a])
    score_b = calculate_score(metrics[idx_b])
    if score_b > score_a:
        idx_a, idx_b = idx_b, idx_a

    prompt = generate_crossover_prompt(
        ref_arch_src=ref_arch_src,
        better_kernel=pool[idx_a],
        better_metrics=metrics[idx_a],
        worse_kernel=pool[idx_b],
        worse_metrics=metrics[idx_b],
        task_params=args.task_params,
    )

    offspring_code, offspring_metrics, logs = propose_step(
        ref_arch_src,
        inference_server,
        kernel_pool=[],
        metrics_pool=[],
        args=args,
        override_prompt=prompt, # bypass pool-prompt construction
    )

    if log_path:
        _save_kernel(log_path, kid, offspring_code, offspring_metrics)
        parent_log = {
            "kid": kid,
            "parent_better": idx_a,
            "parent_worse":  idx_b,
            "parent_better_score": calculate_score(metrics[idx_a]),
            "parent_worse_score":  calculate_score(metrics[idx_b]),
            "offspring_score": calculate_score(offspring_metrics),
        }
        with open(os.path.join(log_path, f"kernel_{kid}_parents.json"), "w") as f:
            json.dump(parent_log, f, indent=2)

    return offspring_code, offspring_metrics, kid


def run_ga_loop(
    ref_arch_src: str,
    inference_server: str,
    args: argparse.Namespace,
    log_path: str | None = None,
) -> tuple[str, EvalResult]:
    """
    Generational GA loop.

    Args (from argparse.Namespace):
        population_size  : fixed pool size (P)
        num_generations  : how many generations to run
        softmax_temperature : temperature for parent selection
        refine_steps     : local refinement steps per offspring (0 = off)
        resume_from      : optional path to resume a previous GA run
    """
    population_size = int(args.population_size)
    num_generations = int(args.num_generations)

    pool:    list[str]            = []
    metrics: list[EvalResult]     = []
    ids:     list[int]            = []
    next_id: int                  = 0
    start_gen: int                = 0

    if getattr(args, "resume_from", None):
        pool, metrics, ids, max_kid = _load_ga_from_logs(
            args.resume_from, population_size
        )
        if pool:
            # Shift `next_id` safely beyond the max ID detected
            next_id = max_kid + 1
            # Infer what generating cycle we are likely supposed to start on
            start_gen = next_id // population_size
            
            if next_id > 0 and log_path and log_path != args.resume_from:
                os.makedirs(log_path, exist_ok=True)
                for kid in ids:
                    for suffix in (".py", "_metrics.json", "_parents.json"):
                        src = os.path.join(args.resume_from, f"kernel_{kid}{suffix}")
                        if os.path.exists(src):
                            shutil.copy2(src, os.path.join(log_path, f"kernel_{kid}{suffix}"))
        else:
            next_id = 0
            start_gen = 0

    if not pool:
        logger.info(f"Seeding initial population of {population_size} kernels...")

        def _seed_one(kid):
            code, m, logs = propose_step(
                ref_arch_src, inference_server,
                [], [], args,
                context_ids=[], elite_kernel_pool=[], elite_metrics_pool=[],
                elite_context_ids=[],
            )
            if log_path:
                _save_kernel(log_path, kid, code, m)
            return code, m, kid

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(_seed_one, i) for i in range(population_size)]
            results = [f.result() for f in tqdm(
                as_completed(futures), total=population_size, desc="Seeding"
            )]

        results.sort(key=lambda r: calculate_score(r[1]), reverse=True)
        pool    = [r[0] for r in results]
        metrics = [r[1] for r in results]
        ids     = [r[2] for r in results]
        next_id = population_size
        start_gen = 1

        _save_generation_log(log_path, 0, list(range(population_size)), ids, metrics)

    for gen in range(start_gen, num_generations + 1):
        logger.info(
            f"Generation {gen}/{num_generations} | "
            f"best={calculate_score(metrics[0])} | pool_size={len(pool)}"
        )

        offspring_ids = list(range(next_id, next_id + population_size))
        next_id += population_size

        # Snapshot pool so all offspring see the same parents
        pool_snapshot    = list(pool)
        metrics_snapshot = list(metrics)

        offspring: list[tuple[str, EvalResult, int]] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(
                    _generate_offspring,
                    ref_arch_src, inference_server,
                    pool_snapshot, metrics_snapshot,
                    args, kid, log_path,
                ): kid
                for kid in offspring_ids
            }
            for f in tqdm(
                as_completed(futures),
                total=population_size,
                desc=f"Gen {gen} offspring",
            ):
                try:
                    offspring.append(f.result())
                except Exception as e:
                    kid = futures[f]
                    logger.error(f"Offspring {kid} failed: {e}")
                    # Insert a null placeholder so pool size arithmetic stays correct
                    offspring.append(("", EvalResult(), kid))

        # Optional local refinement per offspring
        if getattr(args, "refine_steps", 0) > 0:
            from agent.iterative_agent import run_iterative_loop
            refined = []
            for code, m, kid in offspring:
                if code:
                    code, m = run_iterative_loop(
                        ref_arch_src, inference_server, code, m, args,
                        large_loop_id=kid, log_path=log_path,
                    )
                refined.append((code, m, kid))
            offspring = refined

        combined = list(zip(pool, metrics, ids)) + [
            (code, m, kid) for code, m, kid in offspring if code
        ]
        combined.sort(key=lambda t: calculate_score(t[1]), reverse=True)
        combined = combined[:population_size]

        pool    = [t[0] for t in combined]
        metrics = [t[1] for t in combined]
        ids     = [t[2] for t in combined]

        _save_generation_log(
            log_path, gen,
            [kid for _, _, kid in offspring],
            ids, metrics,
        )

        logger.info(
            f"Gen {gen} done | "
            f"best={calculate_score(metrics[0])} | "
            f"survivors={ids}"
        )

    if not pool:
        raise ValueError("GA produced no kernels.")

    return pool[0], metrics[0]