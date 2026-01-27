# Testing Verification Changes

This document summarizes all changes made during the testing guide verification process for the Self-Forcing repository.

## Summary

These changes enable reliable testing of the inference pipeline on systems with NFS-mounted workspaces and limited local storage. The key innovations are:

1. **GPU Memory Mapping** - Load T5 text encoder directly to GPU memory to bypass CPU memory allocation issues
2. **Sequential Loading** - Load models one at a time with garbage collection to reduce memory fragmentation
3. **API Compatibility** - Update inference script for the new pose-conditioning API
4. **Lazy Load Testing** - New test script to verify on-demand pose weight loading

---

## Modified Files

### 1. `inference.py`

**Purpose:** Main inference entry point for video generation.

**Changes:**

| Change | Reason |
|--------|--------|
| `strict=False` in `load_state_dict()` | Allow loading checkpoints that lack new `pose_proj` weights (randomly initialized) |
| Replace `low_memory=low_memory` with `input_image=None, dwpose_data=None, random_ref_dwpose=None` | API updated to require pose/image arguments; None means T2V mode |
| Replace `torchvision.write_video()` with `imageio.mimwrite()` | Fix `pict_type` TypeError in torchvision with newer PyAV versions |

```diff
-    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])
+    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'], strict=False)
```

```diff
-        low_memory=low_memory,
+        input_image=None,
+        dwpose_data=None,
+        random_ref_dwpose=None,
```

```diff
-            write_video(output_path, video[seed_idx], fps=16)
+            import imageio
+            imageio.mimwrite(output_path, video[seed_idx].numpy().astype('uint8'), fps=16)
```

---

### 2. `pipeline/causal_diffusion_inference.py`

**Purpose:** Core inference pipeline for causal video generation.

**Changes:**

| Change | Reason |
|--------|--------|
| Sequential model loading with progress messages | Better debugging and user feedback during startup |
| `gc.collect()` + `torch.cuda.empty_cache()` between loads | Reduce memory fragmentation when loading multiple large models |

```python
# Before: All models loaded in sequence without cleanup
self.generator = WanDiffusionWrapper(...)
self.text_encoder = WanTextEncoder()
self.image_encoder = CLIPModel(...)
self.vae = WanVAEWrapper()

# After: Sequential loading with garbage collection
print("Loading generator...")
self.generator = WanDiffusionWrapper(...)
gc.collect()
torch.cuda.empty_cache()

print("Loading text encoder...")
self.text_encoder = WanTextEncoder()
gc.collect()
torch.cuda.empty_cache()
# ... etc
```

---

### 3. `utils/wan_wrapper.py`

**Purpose:** Wrapper classes for Wan model components (T5, VAE).

**Changes:**

| Change | Reason |
|--------|--------|
| Check `/tmp` first for T5 weights | Use local SSD instead of NFS to avoid memory-mapping errors |
| `map_location='cuda:0'` | Load T5 directly to GPU memory, bypassing problematic CPU allocation |

```diff
-        self.text_encoder.load_state_dict(
-            torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
-                       map_location='cpu', weights_only=False)
-        )
+        import os
+        local_path = "/tmp/models_t5_umt5-xxl-enc-bf16.pth"
+        t5_path = local_path if os.path.exists(local_path) else "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
+        
+        print(f"Loading T5 from {t5_path}...")
+        self.text_encoder.load_state_dict(
+            torch.load(t5_path, map_location='cuda:0', weights_only=False)
+        )
```

---

### 4. `test_load_model.py`

**Purpose:** Level 3 GPU test for model loading and forward pass.

**Changes:**

| Change | Reason |
|--------|--------|
| Check `/dev/shm` first for T5 weights | Prefer RAM disk for fastest loading during tests |
| Real T5 loading instead of mock embeddings | Full validation of text encoder integration |
| Proper cleanup with `del` and cache clear | Avoid memory leaks between test runs |

---

### 5. `test_with_weights.py`

**Purpose:** Level 4 full validation test with real weights.

**Changes:**

| Change | Reason |
|--------|--------|
| Use real `WanTextEncoder` instead of mock | Full validation of text conditioning path |
| Check for `/dev/shm` weights availability | Prefer local copy for reliability |

---

## New Files

### 1. `test_lazy_load.py`

**Purpose:** Verify that pose weights are loaded on-demand (lazy loading).

**Key Features:**
- Creates a dummy pose checkpoint
- Initializes pipeline with `pose_weights_path` pointing to it
- Verifies `pose_weights_loaded` flag is set after first inference call

**Usage:**
```bash
uv run python test_lazy_load.py
```

---

### 2. `configs/tiny_test.yaml`

**Purpose:** Minimal config for fast testing (3 frames instead of 81).

```yaml
independent_first_frame: true
num_output_frames: 3
height: 480
width: 832
num_frames: 3
```

---

### 3. `test_prompts.json`

**Purpose:** Simple single-line prompt file for testing.

```
A person walking in the park
```

---

## Environment Notes

### CLIP Weights

The CLIP model (`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`) is not included in the 1.3B model repository. It must be downloaded from the I2V repository:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Wan-AI/Wan2.1-I2V-14B-480P',
    filename='models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
    local_dir='wan_models/Wan2.1-T2V-1.3B'
)
```

### T5 Local Copy

For reliable loading on NFS systems, copy T5 weights to local storage:

```bash
# Option 1: Local SSD (recommended)
cp wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth /tmp/

# Option 2: RAM disk (faster but uses 11GB RAM)
cp wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth /dev/shm/
```

### Dependencies

New dependencies added during testing:
```bash
uv pip install --system imageio imageio-ffmpeg lmdb
```
