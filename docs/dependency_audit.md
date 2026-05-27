# Dependency Audit

## Scope

This audit covers the dependency files used by the current execution flow:

- `requirements/requirements.txt`
- `requirements/requirements_llama.txt`
- `requirements/requirements_genprm.txt`
- `pyproject.toml`

The classification below is based on repository imports and the runtime path in `scripts/run_local.sh`.

## Kept In `requirements/requirements.txt`

| Package | Why it stays |
| --- | --- |
| `antlr4-python3-runtime` | Required at runtime by OmegaConf/Hydra in `experiments/math/run_vllm.py`. |
| `fire` | Used by `eval/proc_results.py` and `eval/visualize.py`. |
| `graphviz` | Imported by `treesearch.visualization` for tree rendering. |
| `hydra-core` | Used by `experiments/math/run_vllm.py`. |
| `jax`, `jaxlib`, `numpyro`, `pymc`, `xarray`, `packaging`, `pandas` | Required when `treesearch` exposes `ABMCTSM`; imported through `treesearch.algos.ab_mcts_m._ab_mcts_m_imports`. |
| `joblib` | Used by `eval/proc_results.py`. |
| `latex2sympy2-extended`, `lighteval`, `math-verify` | Used by the Math500, AIME24_25, and MinervaMath task evaluators. |
| `matplotlib` | Used by `eval/visualize.py`. |
| `numpy` | Used by `bg_mcts.evaluate_code` and `treesearch` probability code. |
| `omegaconf` | Used by `experiments/math/run_vllm.py`. |
| `openai` | Used by the local vLLM client wrappers and GenPRM client. |
| `pillow` | Imported by prompt and generation interfaces. |
| `pydantic` | Used by `src/bg_mcts/prompts/prompt_configs.py`. |
| `pyyaml` | Required at runtime by OmegaConf/Hydra when loading YAML configuration. |
| `python-dotenv` | Used by `run_vllm.py` and the vLLM wrapper classes. |
| `scipy` | Required by `treesearch` core and `ABMCTSA`. |
| `tqdm` | Used by `run_vllm.py` and `eval/proc_results.py`. |
| `transformers` | Used by the local vLLM wrappers and GenPRM tokenizer loading. |

## Kept In `requirements/requirements_llama.txt`

| Package | Why it stays |
| --- | --- |
| `vllm` | `scripts/run_local.sh` creates the Llama server environment and launches `vllm serve` directly in it. |

## Kept In `requirements/requirements_genprm.txt`

| Package | Why it stays |
| --- | --- |
| `vllm` | `scripts/run_local.sh` creates the GenPRM server environment and launches `vllm serve` directly in it. |

## Optional Or Context-Specific

| Package | Notes |
| --- | --- |
| `pytest` | Only needed for test execution in `treesearch/tests`. |
| `ruff`, `black` | Development tooling only. |
| `jupyter`, `ipywidgets` | Notebook workflow only; not used by current runtime path. |
| `graphviz` system executable | Still required on the host machine for rendering output files; the Python package alone is not enough. |

## Removed As Unused Top-Level Entries

These packages had no direct import/use in the current repository execution flow and are expected to be resolved transitively when needed by retained packages, or were simply unused in this repo:

- `accelerate`
- `anthropic`
- `datasets`
- `huggingface-hub`
- `litellm`
- `networkx`
- `opencv-python-headless`
- `pygraphviz`
- `ray`
- `seaborn`
- `tenacity`
- `vllm` from the main environment
- the large set of notebook, web server, AWS, and formatting packages that were only present through previous environment exports

## Notes

- Versions were preserved exactly as they appeared in the original dependency files for packages that remain.
- `treesearch` is treated as a local in-repo package. The main environment keeps the optional `ABMCTSM` dependencies because `experiments/math/run_vllm.py` imports `treesearch` directly, and `treesearch.__init__` exposes `ABMCTSM` when those dependencies are available.
