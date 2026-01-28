# Self-Forcing Testing Guide

This guide describes how to verify your changes to the `Self-Forcing` repository, specifically the pose and image conditioning features.

> [!NOTE]
> All commands use `uv` for fast dependency management and execution. `test_output.log` is generated during some tests and can be safely ignored or deleted.

## Level 0: Sanity Check (CPU)

Confirm that `uv`, `torch`, and `CUDA` (if applicable) are visible.

**Command:**
```bash
uv run python - <<'PY'
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
PY
```

---

## Level 1: Wiring Test (Fastest - CPU)

Verify that tensor shapes and pipeline connectivity are correct. This can be run on **CPU**.

**Command:**
```bash
uv run python test_wiring.py
```

**What it checks:**
- Pipeline initialization and `einops` shape transformations.
- Batch reshape operations in `CausalWanModel`.

---

## Level 1.5: Pose Alignment (CPU)

Asserts that UniAnimate-style pose frames (81 + 3 prepend) map to 21 latent frames and align with block slicing.

**Command:**
```bash
uv run python test_pose_alignment.py
```

---

## Level 1.6: Train-path Conditioning (GPU)

Verifies that `add_condition` is accepted in the non-cached training forward.

**Command:**
```bash
uv run python test_train_conditioning.py
```

---

## Level 1.8: Pose-only Check (CPU)

Verifies behavior when only `dwpose_data` is provided (no `random_ref_dwpose`).

**Command:**
```bash
uv run python test_pose_only.py
```

---

## Level 2: Minimal Flow (Logic Check - CPU)

Verify end-to-end inference logic using random initializations. This can be run on **CPU**.

**Command:**
```bash
uv run python test_minimal.py
```

**What it checks:**
- Pose embedding forward pass and per-block extraction.
- Data flow through transformer layers without requiring weights.

---

## Level 3: Model Loading (Weights & SDPA Fallback - GPU)

Verify that the 1.3B model and T5 encoder load correctly. This requires a **GPU**. If FlashAttention is not installed, it tests the SDPA fallback path.

**Requirement:** Download weights using `uv` (or `download_models.py`).
```bash
uv run huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
```
Optional one-shot downloader (includes Self-Forcing + ODE + CLIP):
```bash
uv run python download_models.py
```

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python test_load_model.py
```

---

## Level 4: Full System Validation (GPU)

Complete system check validating the new config knobs and component loading.

**Requirement:** Self-Forcing checkpoint.
```bash
uv run huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python test_with_weights.py
```

**Config Knobs Validated:**
- `pose_weights_path`: Path to the UniAnimate pose checkpoint. The lazy loader only triggers when this path is set.
- `pose_weights_strict`: Boolean (defaults to `True`). Controls `state_dict` filtering and load strictness.

---

## Level 4.5: Lazy Pose-Weight Load (Real Checkpoint - GPU)

Verifies that the lazy loader works with a **real** UniAnimate pose checkpoint.

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python test_lazy_load.py --pose-weights-path /path/to/unianimate_pose_checkpoint.pt --strict
```

---

## Level 5: Interactive Testing (GPU)

Run the GUI or CLI to see the actual generated videos.

**GUI Demo:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python demo.py
```

**CLI Inference:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/test_run \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```
