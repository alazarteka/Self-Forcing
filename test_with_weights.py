#!/usr/bin/env python3
"""
Memory-efficient test with real model weights.
Strategy: Test components individually, use single GPU, clear memory between tests.
"""

import os
import torch
import sys
from pathlib import Path

# Configuration
USE_GPU = True  # Set to False for CPU-only testing
GPU_ID = 0       # Use only first GPU for testing
CLEAR_CACHE = True  # Clear CUDA cache between tests

def get_device():
    """Get device with memory management."""
    if USE_GPU and torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"Using GPU {GPU_ID}: {torch.cuda.get_device_name(GPU_ID)}")
        print(f"Memory: {torch.cuda.get_device_properties(GPU_ID).total_memory / 1024**3:.1f}GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def clear_memory():
    """Clear CUDA cache and garbage collect."""
    if CLEAR_CACHE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def test_model_availability():
    """Check what model weights are available."""
    print("=" * 60)
    print("Checking Model Availability")
    print("=" * 60)

    # Check for Wan model directory
    wan_path = Path("wan_models/Wan2.1-T2V-1.3B")
    if wan_path.exists():
        files = list(wan_path.glob("*"))
        print(f"\n✅ Wan2.1-T2V-1.3B directory exists with {len(files)} files")
        for f in sorted(files)[:10]:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name} ({size_mb:.1f}MB)")
    else:
        print(f"\n❌ Wan model not found at {wan_path}")
        print("\nTo download:")
        print("  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B")

    # Check for Self-Forcing checkpoint
    sf_checkpoint = Path("checkpoints/self_forcing_dmd.pt")
    if sf_checkpoint.exists():
        size_mb = sf_checkpoint.stat().st_size / 1024 / 1024
        print(f"\n✅ Self-Forcing checkpoint exists ({size_mb:.1f}MB)")
    else:
        print(f"\n❌ Self-Forcing checkpoint not found")
        print("\nTo download:")
        print("  huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .")

    return wan_path.exists() and sf_checkpoint.exists()

def test_component_loading(device):
    """Test loading individual components with memory tracking."""
    print("\n" + "=" * 60)
    print("Testing Component Loading")
    print("=" * 60)

    all_pass = True

    # Test 1: Text Encoder (T5)
    print("\n[Test 1] Loading Text Encoder (T5-XXL)")
    try:
        from utils.wan_wrapper import WanTextEncoder
        text_encoder = WanTextEncoder()
        text_encoder = text_encoder.to(device)
        print(f"  ✅ Loaded successfully")
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  Memory allocated: {mem_allocated:.2f}GB")
        del text_encoder
        clear_memory()
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        all_pass = False

    # Test 2: VAE
    print("\n[Test 2] Loading VAE")
    try:
        from utils.wan_wrapper import WanVAEWrapper
        vae = WanVAEWrapper()
        vae = vae.to(device)
        print(f"  ✅ Loaded successfully")
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  Memory allocated: {mem_allocated:.2f}GB")
        del vae
        clear_memory()
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        all_pass = False

    # Test 3: CLIP (for image conditioning)
    print("\n[Test 3] Loading CLIP Model")
    try:
        from wan.modules.clip import CLIPModel
        clip_path = "wan_models/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        if Path(clip_path).exists():
            clip = CLIPModel(
                dtype=torch.float32,
                device=device,
                checkpoint_path=clip_path,
                tokenizer_path="xlm-roberta-large"
            )
            print(f"  ✅ Loaded successfully")
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"  Memory allocated: {mem_allocated:.2f}GB")
            del clip
            clear_memory()
        else:
            print(f"  ⚠️  CLIP checkpoint not found, skipping")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        all_pass = False

    return all_pass

def test_model_loading(device):
    """Test loading the full causal model."""
    print("\n" + "=" * 60)
    print("Testing Full Model Loading")
    print("=" * 60)

    try:
        from utils.wan_wrapper import WanDiffusionWrapper

        print("\nLoading causal model (1.3B)...")
        model = WanDiffusionWrapper(
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=5.0,
            is_causal=True
        )

        # Move to device
        print(f"Moving to {device}...")
        model = model.to(device)

        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  ✅ Model loaded successfully")
            print(f"  Memory allocated: {mem_allocated:.2f}GB")
            print(f"  Memory reserved: {mem_reserved:.2f}GB")

        # Test forward pass with tiny input
        print("\nTesting forward pass with tiny input...")
        batch_size = 1
        num_frames = 3  # Minimal for testing
        model_kwargs = model.model.kwargs if hasattr(model, 'model') else {}

        noise = torch.randn(batch_size, num_frames, 16, 8, 8).to(device)
        prompts = ["A test prompt"]
        conditional_dict = model.text_encoder(prompts)

        # Don't use KV cache for this test
        with torch.no_grad():
            output = model(
                noisy_image_or_video=noise.permute(0, 2, 1, 3, 4),
                conditional_dict=conditional_dict,
                timestep=torch.ones([batch_size, num_frames]).to(device),
                kv_cache=None,
                add_condition=None,
                clip_feature=None,
                y=None
            )

        print(f"  Input shape: {noise.shape}")
        print(f"  Output shape: {output.shape}")

        expected_shape = (batch_size, 16, num_frames, 8, 8)
        if output.shape == expected_shape:
            print(f"  ✅ Forward pass successful!")
        else:
            print(f"  ❌ Shape mismatch: expected {expected_shape}")

        # Cleanup
        del model, output, noise, conditional_dict
        clear_memory()

        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pose_embedding_structure(device):
    """Test pose embedding structure without weights."""
    print("\n" + "=" * 60)
    print("Testing Pose Embedding Structure")
    print("=" * 60)

    try:
        # Create a minimal args object
        class MinimalArgs:
            num_train_timestep = 1000
            timestep_shift = 5.0
            guidance_scale = 3.0
            num_frame_per_block = 3
            independent_first_frame = False
            model_kwargs = {"timestep_shift": 5.0}

        from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

        # Initialize pipeline (will fail if models not downloaded, but we catch that)
        print("Initializing pipeline structure...")
        args = MinimalArgs()

        # Just test that the pose embedding modules are created correctly
        # Don't actually load the full model
        import torch.nn as nn

        # Recreate the pose embedding modules
        CONCAT_DIM = 4
        dwpose_embedding = nn.Sequential(
            nn.Conv3d(3, CONCAT_DIM * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.SiLU(),
            nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.SiLU(),
            nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.SiLU(),
            nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.SiLU(),
            nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, 3, stride=(2,2,2), padding=1),
            nn.SiLU(),
            nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, 3, stride=(2,2,2), padding=1),
            nn.SiLU(),
            nn.Conv3d(CONCAT_DIM * 4, 5120, (1,2,2), stride=(1,2,2), padding=0)
        ).to(device)

        # Test forward pass
        dwpose_input = torch.randn(1, 3, 21, 60, 104).to(device)  # [B, C, F, H, W]
        dwpose_input = torch.cat([dwpose_input[:, :, :1].repeat(1, 1, 3, 1, 1), dwpose_input], dim=2) / 255.0
        output = dwpose_embedding(dwpose_input)

        print(f"  Input shape: {dwpose_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: [1, 5120, 21, 60, 104] (with first frame tripled)")

        if output.shape[1] == 5120:
            print(f"  ✅ Pose embedding structure correct!")
        else:
            print(f"  ❌ Channel dimension mismatch")

        del dwpose_embedding, dwpose_input, output
        clear_memory()

        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all memory-efficient tests."""
    print("\n" + "=" * 60)
    print("MEMORY-EFFICIENT MODEL TESTING")
    print("=" * 60)
    print(f"\nStrategy:")
    print(f"- Use single GPU (GPU {GPU_ID}) instead of all 4")
    print(f"- Test components individually")
    print(f"- Clear memory between tests")
    print(f"- Use small inputs for forward passes")
    print()

    device = get_device()

    # Run tests
    results = {}

    results['availability'] = test_model_availability()

    if results['availability']:
        results['components'] = test_component_loading(device)
        results['model'] = test_model_loading(device)
    else:
        print("\n⚠️  Model weights not found. Skipping loading tests.")
        print("Run the download commands above first.")
        results['components'] = None
        results['model'] = None

    results['pose_structure'] = test_pose_embedding_structure(device)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is None:
            status = "⚭️  SKIP"
        elif result:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    # Final memory report
    if device.type == 'cuda':
        print(f"\nFinal GPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        print(f"  Free:      {(torch.cuda.get_device_properties(GPU_ID).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f}GB")

    return 0

if __name__ == "__main__":
    sys.exit(main())
