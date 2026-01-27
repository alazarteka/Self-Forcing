# Project Guidelines

This document captures the scientific goals, current status, and operational workflows for porting UniAnimate‑DiT conditioning into Self‑Forcing and distilling a smaller causal model.

## Scientific Goals

1. **Port UniAnimate‑DiT conditioning into Self‑Forcing**  
   Enable pose and image conditioning inside the causal, KV‑cached inference loop without losing controllability.

2. **Validate inference end‑to‑end**  
   Ensure real‑weight inference works with pose conditioning, lazy pose‑weight loading, and interactive video output.

3. **Distill a high‑quality teacher into a fast student**  
   Teacher: UniAnimate‑DiT 14B with LoRA merged (CFG + pose conditioned, 50×2 steps).  
   Student: Self‑Forcing 1.3B, 4‑step generation, **no LoRA**.

## Current Status

- Pose/image conditioning is integrated in inference and tested.
- Lazy pose‑weight loading validated (`pose_weights_path`).
- Level‑5 interactive inference produces videos.
- Base Wan2.1 key names match between UniAnimate WanModel and Self‑Forcing CausalWanModel (except `pose_proj`).
- Offline LoRA merge script added (`scripts/merge_lora.py`).

## Architecture Snapshot

- **Inference pipeline**: `pipeline/causal_diffusion_inference.py`
  - DWPose embeddings → `add_condition` tokens
  - Image conditioning via CLIP + VAE
  - KV cache + blockwise causal attention
- **Core model**: `wan/modules/causal_model.py`
  - `pose_proj` maps 5120 → model dim for 1.3B
  - `add_condition` injected after patch embedding
- **Wrappers**: `utils/wan_wrapper.py`
  - T5 text encoder, VAE wrapper, model wrapper

## Testing Ladder

See `testing_guide.md` for the full ladder. Key commands:
```bash
uv run python test_wiring.py
uv run python test_pose_alignment.py
uv run python test_pose_only.py
uv run python test_minimal.py
CUDA_VISIBLE_DEVICES=0 uv run python test_load_model.py
CUDA_VISIBLE_DEVICES=0 uv run python test_with_weights.py
CUDA_VISIBLE_DEVICES=0 uv run python test_lazy_load.py --pose-weights-path /path/to/pose_ckpt.pt --strict
```

## Weights & Storage Notes

- Wan2.1 base weights: `Wan-AI/Wan2.1-T2V-1.3B`
- CLIP weights: `Wan-AI/Wan2.1-I2V-14B-480P`
- Self‑Forcing checkpoint: `gdhe17/Self-Forcing`
- Use `download_models.py` to fetch all required assets.

T5 loading uses `map_location='cuda:0'` to avoid NFS mmap issues.  
For local reliability, copy the T5 checkpoint to `/tmp` (or `/dev/shm`).

## Offline LoRA Merge (Teacher Prep)

Keep LoRAs separate and merge **only when preparing the teacher**:
```bash
uv run python scripts/merge_lora.py \
  --base /path/to/Wan2.1-14B-base.safetensors \
  --lora /path/to/UniAnimate-14B-lora.pt \
  --out /path/to/Wan2.1-14B-merged.safetensors \
  --alpha 1.0 \
  --device cpu \
  --dtype float32
```

## Distillation Plan (High‑Level)

1. Merge LoRA into 14B teacher (offline).
2. Run CFG+pose‑conditioned inference at 50×2 steps to generate teacher targets.
3. Train Self‑Forcing 1.3B to match teacher outputs in 4 steps.
4. Validate pose adherence and temporal coherence.

## Open Work

- Training integration for pose conditioning (teacher‑forcing path).
- Pose‑only behavior choice (currently gated off unless both pose inputs exist).
- Formal evaluation metrics for pose fidelity and temporal stability.
