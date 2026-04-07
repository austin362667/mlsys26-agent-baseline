
import re

PROBLEM_STATEMENT = """You write custom GPU kernels that outperform reference PyTorch operators.
Analyze bottlenecks via arithmetic intensity, memory access patterns, and FLOP utilization.
"""

TRITON_PROMPT = """Generate a Triton 3.3.1 kernel optimized for {target_gpu} that implements:

{definition}

Reference PyTorch implementation (functionally correct but unoptimized — replace it entirely):
```python
{ref_arch_src}
```

Requirements:
- Clean, efficient Triton code targeting {target_gpu} architecture
- Modern Triton syntax (imports: torch, triton, triton.language as tl)
- Match the reference implementation's numerical accuracy; optimize for performance
- Use tensor shapes, dtypes, and axes from the definition to guide memory access and tiling
- The reference exposes the mathematical spec and I/O contract — your kernel must satisfy the same interface

The wrapper function must:
- Accept the same arguments as the reference `run()` function
- Move CPU tensors to GPU (.cuda()) if CUDA is available; raise clearly if not
- Restore outputs to the original input device
- Expose a `run` entry point with identical signature

Constraints:
- Valid Python only — no hex float literals, no C/CUDA syntax
- Return only runnable code, no markdown or explanations
"""

CROSSOVER_PROMPT = """## Genetic Crossover

Reason step-by-step to generate a correct and efficient offspring kernel.
Recombines the best parts of both Parent A and Parent B kernels, while avoiding their pitfalls.

---

### Parent A
{better_status}
```python
{better_kernel}
```

---

### Parent B
{worse_status}
```python
{worse_kernel}
```

---

### Instructions

Think step-by-step before writing code:

1. **Diagnose each parent**
   - Compile errors: identify root cause; avoid repeating them
   - Correctness errors: trace the numerical issue; do not replicate it
   - Performance: which workloads is each parent weak on, and why?

2. **Extract the best ideas** from each parent — tiling strategy, memory layout, vectorization width, loop ordering, shared memory usage

3. **Add one deliberate mutation** beyond what either parent does, justified by the per-workload bottlenecks

4. Output ONLY the final kernel code — no explanation, no markdown fences

5. Also use a helper function to enqueue the above kernel with appropriate grid/block sizes.
   Follow the signature matching the reference implementation's I/O contract. 
    ```python
    import torch

    @torch.no_grad()
    def run(
        routing_logits: torch.Tensor,
        routing_bias: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        local_expert_offset: int,
        routed_scaling_factor: float,
    ):
        
        \"\"\"
        FP8 block-scale dequantization: float ≈ fp8 * scale
        DeepSeek-V3 no-aux routing:
            s = sigmoid(logits)
            s_with_bias = s + bias
            group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
            on the kept groups, take global top_k=8 experts
            combine with weights derived from s (without bias), normalized and
            scaled by routed_scaling_factor
        Local computation:
            only experts in [local_expert_offset, local_expert_offset + E_local) are
            computed on this rank (GEMM1 → SwiGLU → GEMM2), then per-token weighted
            accumulation.
        \"\"\"
    ```
"""

def _format_metrics(metrics: "EvalResult | None", label: str) -> str:
    if metrics is None:
        return f"[{label}] Not evaluated"
    if not metrics.compiled:
        return f"[{label}] COMPILE ERROR\n{metrics.error or 'unknown'}"
    if not metrics.correct:
        return f"[{label}] \n{metrics.error or 'unknown'}"

    lines = [
        f"[{label}] CORRECT",
        f"Geomean speedup: {metrics.speedup:.4f}x",
    ]
    if metrics.speedup_per_workload:
        per_wl = "\n".join(f"  {wid}: {s:.4f}x" for wid, s in metrics.speedup_per_workload)
        lines.append(f"Per-workload:\n{per_wl}")
    if metrics.latency_ms is not None:
        lines.append(f"Avg latency: {metrics.latency_ms:.3f} ms")
    return "\n".join(lines)


def generate_crossover_prompt(
    ref_arch_src: str,
    better_kernel: str,
    better_metrics: "EvalResult | None",
    worse_kernel: str,
    worse_metrics: "EvalResult | None",
    task_params: dict,
) -> str:
    required = set(re.findall(r"\{(\w+)\}", TRITON_PROMPT)) - {"ref_arch_src"}
    missing = required - task_params.keys()
    if missing:
        raise ValueError(f"Missing prompt parameters: {missing}")

    task_header = PROBLEM_STATEMENT + TRITON_PROMPT.format(
        ref_arch_src=ref_arch_src, 
        **task_params
    )
    
    crossover_body = CROSSOVER_PROMPT.format(
        better_status=_format_metrics(better_metrics, "Parent A"),
        better_kernel=better_kernel,
        worse_status=_format_metrics(worse_metrics, "Parent B"),
        worse_kernel=worse_kernel,
    )
    print("Generated crossover prompt:")
    print(task_header + "\n\n" + crossover_body + '\n\n')
    return task_header + crossover_body