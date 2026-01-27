# AGENTS.md

## Mission

Port UniAnimate-DiT pose/image conditioning into the Self-Forcing (causal) pipeline, and validate that inference works end-to-end with real weights. Training integration is planned but not fully implemented yet.

## Current State

- **Inference wiring:** Pose and image conditioning are integrated in:
  - `pipeline/causal_diffusion_inference.py`
  - `wan/modules/causal_model.py`
  - `utils/wan_wrapper.py`
- **Pose CNNs:** `dwpose_embedding` (3D CNN) and `randomref_embedding_pose` (2D CNN) are ported and can load pretrained weights lazily.
- **Projection:** 5120 → model dim via `pose_proj` for the 1.3B model.
- **Lazy pose weights:** `pose_weights_path` triggers on-demand loading; validated by `test_lazy_load.py`.
- **Pose alignment:** Assertion enforces that pose embedding frames match the full output timeline.

## Repos and References

- Self-Forcing (this repo)
- UniAnimate-DiT (sibling): `../UniAnimate-DiT`
- DWPose (sibling): `../dwpose`

Key reference code in UniAnimate:
- `diffsynth/pipelines/wan_video.py`
- `diffsynth/models/wan_video_dit.py`

## Testing

Use the ladder in `testing_guide.md`. Key scripts:
- `test_wiring.py`, `test_minimal.py`
- `test_load_model.py`, `test_with_weights.py`
- `test_lazy_load.py` (supports real checkpoint with `--pose-weights-path`)

For fast end-to-end runs:
- `configs/tiny_test.yaml`
- `test_prompts.json`

## Known Caveats

- **Pose-only behavior:** Currently gated off unless `dwpose_data` and `random_ref_dwpose` are both present.
- **T5 loading:** Uses `map_location='cuda:0'` and checks `/tmp` first to avoid NFS mmap issues. CPU-only runs will fail unless adjusted.
- **CLIP weights:** Pulled from the I2V repo (see `download_models.py`).

## Next Steps (Open Work)

1. **Training integration**
   - Thread `add_condition` through the teacher-forcing path in `utils/wan_wrapper.py`.
   - Extend datasets to supply `dwpose_data` and `random_ref_dwpose`.
2. **Pose-only support**
   - Allow `dwpose_data` without `random_ref_dwpose` if desired.
3. **Real pose checkpoint validation**
   - Use `test_lazy_load.py --pose-weights-path /path/to/pose_ckpt.pt`.

## Where to Start

- Read `CHANGES.md` for recent fixes.
- Run `uv run python test_lazy_load.py --pose-weights-path ...` for real lazy-load validation.
- Compare with UniAnimate’s conditioning flow to confirm parity.
