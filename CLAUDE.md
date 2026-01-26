# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-Forcing is a research implementation of autoregressive video diffusion models that bridges the train-test gap by simulating inference during training using KV caching and autoregressive rollout. This enables real-time, streaming video generation on a single RTX 4090.

## Installation and Setup

```bash
# Create environment
conda create -n self_forcing python=3.10 -y
conda activate self_forcing

# Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Common Commands

### Running Inference

CLI inference (recommended for batch processing):
```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```

GUI demo (for interactive use):
```bash
python demo.py
```

### Training

The training is data-free (no video data needed). First download initialization checkpoint and prompts:
```bash
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```

Distributed training (64 H100 GPUs example):
```bash
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb
```

### Model Download

```bash
# Base model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B

# Self-Forcing checkpoint
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

## Architecture Overview

### Core Innovation

The project implements **causal autoregressive video generation** where:
- Training simulates inference using KV caching (the "self-forcing" technique)
- This resolves train-test distribution mismatch in autoregressive models
- Enables streaming generation with constant memory usage per frame

### Key Components

1. **CausalDiffusionInferencePipeline** (`pipeline/causal_diffusion_inference.py`)
   - Main inference pipeline orchestrating video generation
   - Manages KV cache state across autoregressive steps
   - **Porting features from UniAnimate-DiT**:
     - `dwpose_embedding`: 3D CNN for pose sequence conditioning (CONCAT_DIM=4 → 5120 output)
     - `randomref_embedding_pose`: 2D CNN for reference pose (CONCAT_DIM=4 → RANDOMREF_DIM=20 output)
     - `encode_image()`: Image preprocessing and CLIP/VAE encoding
     - `load_pose_embedding_weights()`: Load pretrained pose embedding weights
     - New inference params: `input_image`, `dwpose_data`, `random_ref_dwpose`

2. **CausalWanModel** (`wan/modules/causal_model.py`)
   - Core transformer with block-wise causal attention masking
   - Implements KV caching for efficient autoregressive inference
   - Uses flex_attention compiled with "max-autotune-no-cudagraphs" for the 1.3B model
   - **Porting addition**: `add_condition` parameter for token-level conditioning
     - Implementation: `x = x + add_condition` in forward() before token concatenation
     - Validates shape matches: `[B, L, C]`
   - Key parameters:
     - `num_frame_per_block`: Frames processed per block (default: 3)
     - `local_attn_size`: Controls attention window size (-1 for full attention)
     - `sink_size`: Number of sink tokens at sequence start

3. **WanDiffusionWrapper** (`utils/wan_wrapper.py`)
   - Wrapper around CausalWanModel handling flow matching predictions
   - Manages both causal and non-causal inference modes
   - Integrates with flow matching schedulers (UniPC/DPM++)
   - **Porting addition**: `add_condition` parameter threaded through to `CausalWanModel`

4. **Model Wrappers** (`utils/wan_wrapper.py`)
   - **WanTextEncoder**: T5-XXL text encoder for prompt embeddings
   - **WanVAEWrapper**: Video VAE for latent space encoding/decoding
   - **CLIPModel**: Image conditioning features

### Data Flow

```
Text Prompts → WanTextEncoder → Text Embeddings
     ↓
Images (optional) → CLIPModel + VAE → Image Embeddings + Latent
     ↓
Pose Data (optional) → dwpose_embedding → Condition Tokens
     ↓
CausalWanModel with KV Cache → Flow Predictions (autoregressive)
     ↓
Scheduler (FlowUniPC/FlowDPM++) → Video Latents
     ↓
WanVAEWrapper → Pixel Space Video
```

### Attention Mechanism

The model uses a sophisticated causal attention pattern:
- **Block-wise causal**: Each frame block attends to all previous blocks
- **Local attention**: Optional windowed attention within blocks (controlled by `local_attn_size`)
- **Sink tokens**: Special tokens at sequence start that all frames attend to
- **Rotary Positional Embeddings**: Applied spatiotemporally with `causal_rope_apply`

### Configuration System

Config files use YAML (see `configs/` directory):
- `self_forcing_dmd.yaml`: DMD (Distribution Matching) training config
- `self_forcing_sid.yaml`: Score Distillation training config
- `default_config.yaml`: Base configuration

Key config parameters:
- `num_frame_per_block`: Frames per autoregressive step (default: 3)
- `timestep_shift`: Flow matching timestep shifting (default: 5.0)
- `guidance_scale`: Classifier-free guidance strength
- `denoising_loss_type`: Loss function (flow, dmd, sid)
- `model_kwargs`: Model-specific parameters passed to WanDiffusionWrapper

### Training Architecture

Multiple trainer implementations available:
- **DiffusionTrainer**: Standard diffusion training
- **GANTrainer**: Adversarial training with discriminator
- **ODETrainer**: ODE-based initialization
- **ScoreDistillationTrainer**: Score distillation sampling

All training is distributed using PyTorch FSDP with `hybrid_full` sharding strategy.

## Integration Notes

### Model Dimension Mismatch (Critical)

**Important:** There's a channel dimension mismatch between the pose embedding and the 1.3B model:

| Model | `dim` (channels) | Pose Embedding Output |
|-------|-----------------|----------------------|
| 1.3B | **1536** | 5120 (from UniAnimate) |
| 14B | **5120** | 5120 (from UniAnimate) ✅ |

The UniAnimate-DiT pose embedding (`dwpose_embedding`) outputs **5120 channels**, designed for the 14B model. When using the 1.3B model (`dim=1536`), a projection layer is required.

**Solution implemented:** A projection layer in `CausalWanModel` maps 5120 → 1536:
```python
self.pose_proj = nn.Linear(5120, self.dim)  # 5120 → 1536
```

**To use 14B model instead** (no projection needed):
```python
WanDiffusionWrapper(model_name="Wan2.1-T2V-14B", is_causal=True)
```

### UniAnimate-DiT Porting Effort

This codebase is actively porting pose and image conditioning features from `../UniAnimate-DiT` to enable fast inference with Self-Forcing's causal architecture.

**What's being ported:**
- **Pose conditioning**: DWPose-based whole-body pose control
- **Image conditioning**: Reference image conditioning with CLIP features
- **Fast inference**: TeaCache, USP, and other acceleration techniques

**Implementation status:**

1. **Pose Conditioning** (Partially implemented)
   - `dwpose_embedding`: 3D CNN processing pose sequences → 5120-dim embeddings
   - `randomref_embedding_pose`: 2D CNN for reference pose → 20-dim embeddings
   - Integration point: Added to tokens via `add_condition` parameter in `CausalWanModel.forward()`
   - Implementation: `x = x + add_condition` (token-level conditioning)
   - TODO: Complete CLIP model weight loading (currently placeholder paths)

2. **Image Conditioning** (Skeleton only)
   - `encode_image()` method added to `CausalDiffusionInferencePipeline`
   - Uses CLIP visual encoder + VAE for image embeddings
   - Returns `{"clip_feature": ..., "y": ...}` dict format
   - TODO: Wire image embeddings into the actual model forward pass
   - Currently marked as "not wired yet" with placeholder dict

3. **Inference Interface Changes**
   - New parameters added to `inference()`:
     - `input_image`: Reference image for conditioning
     - `dwpose_data`: Pose sequence data
     - `random_ref_dwpose`: Reference pose frame
   - Pose processing happens per-block in the denoising loop
   - Condition embeddings extracted with `rearrange` and passed as `add_condition`

**Key differences from UniAnimate-DiT:**
- Original: Processes full sequence at once with standard attention
- Self-Forcing: Block-wise causal processing with KV caching
- Challenge: Adapting per-frame conditioning to block-wise autoregressive generation

### KV Caching

The KV cache implementation:
- Maintains separate caches for positive/negative guidance flows
- Cross-attention caches for text conditioning
- Cache position tracking via `kv_cache_pos` and `kv_cache_neg`
- Automatically manages cache invalidation between generations

## Performance Considerations

- **Memory**: 24GB GPU minimum (tested on RTX 4090, A100, H100)
- **Speed optimization**: Enable `torch.compile`, use TAEHV-VAE, or FP8 layers
- **Distributed training**: Supports gradient accumulation for smaller GPU counts
- **Flex attention**: Requires "max-autotune-no-cudagraphs" mode for 1.3B model due to head configuration

## Important Implementation Details

1. **Flexible attention compilation**: The `flex_attention` call is compiled globally at module level in `causal_model.py` - this is intentional for the 1.3B model's specific head configuration.

2. **Frame ordering**: The model expects inputs in `[batch, frames, channels, height, width]` format but internally permutes between different orderings for different components.

3. **VAE caching**: The VAE decoder supports frame-by-frame caching via `use_cache=True` for streaming decode.

4. **Prompt length**: The model performs better with long, detailed prompts (use prompt extension or LLMs like GPT-4o to expand short prompts).

5. **Pose conditioning integration** (from UniAnimate-DiT port):
   - **Projection layer**: `CausalWanModel` has `pose_proj` layer to map 5120 → model_dim
     - For 1.3B model: `nn.Linear(5120, 1536)` projects pose embedding channels to match model
     - For 14B model: `nn.Identity()` (no projection needed, dimensions match)
   - **Shape handling**: Model reshapes `x` from `[B*L, C]` → `[B, L, C]` before adding condition
   - Per-block pose extraction in denoising loop with `rearrange`: `'b c f h w -> b (f h w) c'`
   - Passed through pipeline: `inference()` → `generator()` → `model.forward()`
   - Only positive flow gets conditioning (`add_condition`), negative flow gets `None` for classifier-free guidance
   - Weights loaded via `load_pose_embedding_weights()` which filters state_dict by prefix

6. **TODOs for UniAnimate-DiT port**:
   - Download CLIP model checkpoint for image conditioning:
     - File: `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
     - Location: `wan_models/Wan2.1-T2V-1.3B/`
     - Source: Available in Wan2.1 I2V model downloads
   - Download pose embedding weights (dwpose, randomref) from UniAnimate-DiT
   - Test the full pipeline with real pose/image data
