# MLSys 2026 FlashInfer AI Kernel Generation Contest: Agent Baseline

An LLM agent baseline for the [MLSys 2026 FlashInfer AI Kernel Generation Contest](https://mlsys26.flashinfer.ai/). See the [flashinfer-bench-starter-kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit) to get started.

An LLM agent baseline that iteratively generates and refines Triton kernels for high-performance LLM operations on NVIDIA GPUs, evaluated via [FlashInfer-Bench](https://bench.flashinfer.ai). For the benchmarking framework code, see the [flashinfer-bench repo](https://github.com/flashinfer-ai/flashinfer-bench/).

## Project Structure

```
agent/
  main.py              # Entry point & task orchestration
  iterative_agent.py   # Iterative Agent: propose + refine loop
  evolve_agent.py      # Evolve Agent: elite pool evolution loop
  ga_agent.py          # Genetic Algorithm Agent: select best and recombine parts of parent kernels and evolve children kernels loop
  api.py               # LLM API client (OpenAI / Claude)
  eval.py              # Kernel evaluation via flashinfer-bench API
  modal_eval.py        # Remote kernel evaluation on Modal GPU
  utils.py             # Shared utilities & data helpers
prompt/
  tuner_prompt.py      # Kernel tuning prompt (str_replace edits)
  proposer_prompt.py   # Kernel proposal prompt
  ga_prompt.py          # GA cross-over prompt
config/
  config_iterative.yaml   # Iterative agent config
  config_evolve.yaml      # Evolve agent config
  config_ga.yaml          # GA agent confgi
  config_mini_test.yaml   # Quick smoke test config
  tasks_default.txt    # Default task list
  tasks_mini.txt       # Minimal task list for smoke test
datasets/              # FlashInfer-Trace / MLSys contest datasets
requirements.txt       # Python dependencies
```

## Quick Start

### 0. Install Dependencies

```bash
pip install -r requirements.txt
```

### 1. Download the Dataset

```bash
mkdir datasets
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest datasets/mlsys26-contest
```

### 2. Set API Key

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
```

### 3. Run the Agent

**Local GPU:**

```bash
python3 -m agent.main --config config/config_mini_test.yaml
```

**Remote GPU via [Modal](https://modal.com/) (no local GPU needed):**

```bash
pip install modal
modal setup  # one-time auth

python3 -m agent.main --config config/config_mini_test.yaml \
  --eval_backend modal --modal_gpu B200
```

The dataset is automatically uploaded to a Modal Volume on first run and cached for subsequent runs.

## Agent Types

| Type | Description |
|------|-------------|
| **iterative** | Proposes an initial kernel, then repeatedly tunes it via str_replace edits |
| **evolve** | Proposes multiple kernels, maintains a recent + elite pool, samples and evolves |

## Config

Example (`config/config_iterative.yaml`):

```yaml
test_source: mlsys26-contest
agent_type: iterative
tasks_path: config/tasks_default.txt
gpu_name: B200
gpu_architecture: Blackwell
api_type: claude
model_name: claude-sonnet-4-5
total_steps: 25
eval_backend: local     # "local" or "modal"
modal_gpu: B200         # GPU type for Modal (ignored when eval_backend=local)
```

Available configs:

| Config | Agent Type |
|--------|------------|
| `config_iterative.yaml` | Iterative Agent |
| `config_evolve.yaml` | Evolve Agent |
| `config_mini_test.yaml` | Quick smoke test |

Key parameters:

- `test_source`: `mlsys26-contest` or `flashinfer-trace`
- `agent_type`: `iterative` or `evolve`
- `tasks_path`: file listing op types / problem IDs to solve
- `total_steps`: number of iterations per task
- `api_type`: `openai` or `claude`
- `model_name`: LLM model to use
- `eval_backend`: `local` (default) or `modal` for remote GPU evaluation
- `modal_gpu`: GPU type on Modal (e.g. `B200`)

### Task List Format

One op type per line. Optionally specify kernel definition IDs after the op type:

```
dsa_paged
gdn
moe
gemm gemm_n128_k2048, gemm_n256_k4096
```

If no kernel definition IDs are given, all kernel definitions under that op type are loaded.

## Output

Results are saved under `outputs/`:

```
outputs/<agent_type>_<test_source>_<steps>_<timestamp>/
  config.yaml
  <op_type>_<problem_id>/
    reference_src.py
    proposal_0_1.py / tune_0_2.py / ...
    global_best_kernel_25.py
    global_best_metrics_25.json
```

## Resume

```bash
python3 -m agent.main \
  --config config/config_iterative.yaml \
  --resume_from outputs/iterative_mlsys26-contest_25_20260208-121400
```

Tasks with existing results are skipped; incomplete tasks continue from where they left off.

## License

See [LICENSE](LICENSE).
