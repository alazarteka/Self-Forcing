# Repository Guidelines

## Project Structure & Module Organization

- `pipeline/` holds inference pipelines (e.g., `causal_diffusion_inference.py`).
- `wan/` contains core model components and attention blocks.
- `utils/` provides wrappers (text encoder, VAE) and helpers.
- `model/` and `trainer/` implement training and loss logic.
- `configs/` stores YAML configs; `configs/tiny_test.yaml` is for quick runs.
- Tests live at repo root: `test_*.py` and `testing_guide.md`.
- Assets: prompts in `prompts/`, output videos in `videos/`.

## Build, Test, and Development Commands

- Install dependencies: `uv pip install --system -r requirements.txt`
- Download weights: `uv run python download_models.py`
- Run wiring tests (CPU): `uv run python test_wiring.py`
- Minimal flow (CPU/GPU): `uv run python test_minimal.py`
- GPU load checks: `CUDA_VISIBLE_DEVICES=0 uv run python test_load_model.py`
- Lazy pose load: `CUDA_VISIBLE_DEVICES=0 uv run python test_lazy_load.py --pose-weights-path /path/to/pose_ckpt.pt --strict`
- Inference: `CUDA_VISIBLE_DEVICES=0 uv run python inference.py --config_path configs/self_forcing_dmd.yaml ...`

## Coding Style & Naming Conventions

- Python, 4‑space indentation, standard PEP8 style.
- Keep new scripts in `scripts/` and name with verbs (e.g., `merge_lora.py`).
- Test files follow `test_*.py` naming.

## Testing Guidelines

- See `testing_guide.md` for the full ladder.
- Prefer CPU tests for wiring; use GPU tests for real weights.
- Tests are standalone scripts; no pytest harness.

## Commit & Pull Request Guidelines

- Commit messages in this repo are short, imperative, and sentence‑case (e.g., “Port UniAnimate-DiT pose…”).
- PRs should include: summary of changes, test commands run, and any artifacts (e.g., sample videos or logs).

## Architecture & Research Notes

- Goal: port UniAnimate‑DiT conditioning into Self‑Forcing’s causal pipeline and distill to a LoRA‑free 1.3B student.
- Pose/image conditioning lives in `pipeline/` and `wan/modules/causal_model.py`.
- Offline LoRA merge is supported via `scripts/merge_lora.py` before distillation.
