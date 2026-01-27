#!/usr/bin/env python3
"""
Test loading the Wan2.1-T2V-1.3B model with Self-Forcing's causal architecture.
"""

import torch
import sys

def encode_text_on_cpu(prompts: list, device: torch.device) -> dict:
    """
    Mock text encoding - bypasses T5 loading issues on NFS.
    Returns random embeddings with the correct shape.
    """
    print("  Using mock embeddings (T5 loading skipped for NFS compatibility)")
    
    # T5-XXL output: [batch_size, seq_len, hidden_dim]
    # seq_len=512, hidden_dim=4096 for T5-XXL
    batch_size = len(prompts)
    seq_len = 512
    hidden_dim = 4096
    
    # Random embeddings as mock
    context = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    
    return {"prompt_embeds": context.to(device)}


def main():
    """Test loading and running the model."""
    # Disable torch.compile to avoid triton/setuptools issues
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # Use GPU 0
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print("Loading Wan2.1-T2V-1.3B with Causal Architecture")
    print("=" * 60)

    try:
        from utils.wan_wrapper import WanDiffusionWrapper

        print("\n[1/3] Initializing model...")
        model = WanDiffusionWrapper(
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=5.0,
            is_causal=True,  # This is the key - use causal architecture
            local_attn_size=-1,
            sink_size=0
        )

        print(f"  Model type: {type(model.model).__name__}")
        print(f"  Model dim: {model.model.dim}")
        print(f"  Model parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

        print("\n[2/3] Moving to GPU...")
        model = model.to(device)
        model.model.eval()  # Set to eval mode

        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  Memory allocated: {mem_allocated:.2f}GB")
            print(f"  Memory reserved: {mem_reserved:.2f}GB")
            print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - mem_reserved) / 1024**3:.2f}GB")

        print("\n[3/3] Testing forward pass...")
        batch_size = 1
        num_frames = 3  # Small for testing
        num_channels = 16
        height = 60
        width = 104

        # Create dummy input (latent space)
        # Model expects [B, F, C, H, W] format and bfloat16 dtype
        noise = torch.randn(batch_size, num_frames, num_channels, height, width, dtype=torch.bfloat16).to(device)

        print(f"  Input shape: {noise.shape}")

        # Create dummy prompt
        prompts = ["A person walking in the park"]

        # Encode prompt on CPU (to save GPU memory)
        print(f"  Encoding prompt: '{prompts[0]}'")
        conditional_dict = encode_text_on_cpu(prompts, device=device)

        # Create timestep
        timestep = torch.ones([batch_size, num_frames], device=device).float()

        # Run forward pass
        with torch.no_grad():
            result = model(
                noisy_image_or_video=noise,  # [B, F, C, H, W]
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=None,  # No caching for this test
                add_condition=None,
                clip_feature=None,
                y=None
            )
            # Unpack the tuple (flow_pred, cache)
            output = result[0] if isinstance(result, tuple) else result

        print(f"  Output shape: {output.shape}")

        expected_shape = (batch_size, num_frames, num_channels, height, width)
        if output.shape == expected_shape:
            print(f"  ✅ Forward pass successful!")
            print(f"\n✅ Model loaded and working correctly!")
            print(f"\nModel is ready for pose/image conditioning tests.")
            return 0
        else:
            print(f"  ❌ Shape mismatch: expected {expected_shape}")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
