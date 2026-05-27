# Repo Cleanup Inventory

This file records the current cleanup classification for the math-focused Bg-MCTS repository.

## Keep

- `README.md`
- `benchmarks/`
- `experiments/math/`
- `src/bg_mcts/`
- `treesearch/src/treesearch/` 
- `scripts/run_local.sh`
- `scripts/background/local_experiments.sh`
- `scripts/eval_math.sh`
- `scripts/check_experiments.sh`
- `requirements/requirements.txt`
- `requirements/requirements_check.txt`
- `requirements/requirements_genprm.txt`
- `requirements/requirements_llama.txt`

## Local Artifact

- `.env`
- `logs/`
- `outputs/`
- `__pycache__/` directories under `experiments/math/` and `src/bg_mcts/`
- `.pyc` files generated during local runs

These are intentionally kept in the working repository for now, but they should not be treated as publishable source files.

## Legacy

- None identified in the current tracked tree.

The remaining shell entrypoints are still part of the active workflow:

- `scripts/run_local.sh` launches local experiments
- `scripts/background/local_experiments.sh` is called from `scripts/run_local.sh`
- `scripts/eval_math.sh` processes and visualizes result directories
- `scripts/check_experiments.sh` checks experiment progress

## Delete Candidate

- `docs/dependency_audit.md`
  - Keep only if you want to publish internal dependency notes.
- Environment YAML files if reintroduced outside the current tracked tree
  - Prefer the `requirements*.txt` files as the maintained environment entrypoints.
