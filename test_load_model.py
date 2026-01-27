#!/usr/bin/env python3
"""
Test loading the Wan2.1-T2V-1.3B model with Self-Forcing's causal architecture.
"""

import torch
import sys

def encode_text_on_cpu(prompts, device=None):
    """Encode text prompts on CPU to save GPU memory."""
    from wan.modules.tokenizers import HuggingfaceTokenizer
    from wan.modules.t5 import umt5_xxl

    cpu_device = torch.device('cpu')

    # Load T5 text encoder on CPU
    text_encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.float32,
        device=cpu_device
    ).eval().requires_grad_(False)
    text_encoder.load_state_dict(
        torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                   map_location=cpu_device, weights_only=False)
    )

    # Load tokenizer
    tokenizer = HuggingfaceTokenizer(
        name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/",
        seq_len=512, clean='whitespace'
    )

    # Tokenize
    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(cpu_device)
    mask = mask.to(cpu_device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    # Encode
    context = text_encoder(ids, mask)

    # Zero out padding
    for u, v in zip(context, seq_lens):
        u[v:] = 0.0

    # Move to target device if specified
    if device is not None:
        context = context.to(device)

    # Convert to bfloat16 to match model dtype
    context = context.to(torch.bfloat16)

    return {"prompt_embeds": context}


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
