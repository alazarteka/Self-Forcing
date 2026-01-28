#!/usr/bin/env python3
"""
Level 1.6: Train-path conditioning smoke check (GPU).
Verifies that add_condition is accepted in the non-cached training forward.
"""

import sys
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for this test.")
        return 1

    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # Small shape to keep the test light
    batch_size = 1
    num_frames = 3
    height = 104
    width = 60
    num_channels = 16

    model = WanDiffusionWrapper(is_causal=True).to(device)
    model.model.eval()

    text_encoder = WanTextEncoder()
    prompts = ["A person walking in the park"]
    conditional_dict = text_encoder(prompts)

    noise = torch.randn(
        batch_size, num_frames, num_channels, height, width, device=device
    )
    timestep = torch.ones([batch_size, num_frames], device=device).float()

    # Build add_condition with matching token length
    with torch.no_grad():
        sample = noise.permute(0, 2, 1, 3, 4)[0]  # [C, F, H, W]
        patch = model.model.patch_embedding(sample.unsqueeze(0))
        token_len = patch.flatten(2).transpose(1, 2).shape[1]
    add_condition = torch.randn(batch_size, token_len, 5120, device=device)
    conditional_dict = dict(conditional_dict)
    conditional_dict["add_condition"] = add_condition

    with torch.no_grad():
        output = model(
            noisy_image_or_video=noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=None
        )

    expected = (batch_size, num_frames, num_channels, height, width)
    print(f"Output shape: {output.shape}")
    if output.shape != expected:
        print(f"❌ Shape mismatch: expected {expected}")
        return 1

    print("✅ Train-path conditioning check passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
