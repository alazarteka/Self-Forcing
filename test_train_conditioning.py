#!/usr/bin/env python3
"""
Level 1.6: Train-path conditioning smoke check (GPU).
Verifies that add_condition is accepted in the non-cached training forward.
"""

import sys
import torch

from utils.wan_wrapper import WanDiffusionWrapper


class MockTextEncoder:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, text_prompts):
        # Match WanTextEncoder output structure: prompt_embeds [B, 512, 4096]
        batch = len(text_prompts)
        return {
            "prompt_embeds": torch.randn(
                batch, 512, 4096, device=self.device, dtype=self.dtype
            )
        }


def main():
    # Disable torch.compile to avoid triton/setuptools issues
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

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
    param_dtype = next(model.parameters()).dtype

    prompts = ["A person walking in the park"]
    conditional_dict = MockTextEncoder(device=device, dtype=param_dtype)(prompts)

    noise = torch.randn(
        batch_size,
        num_frames,
        num_channels,
        height,
        width,
        device=device,
        dtype=param_dtype,
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

    if isinstance(output, tuple):
        flow_pred = output[0]
    else:
        flow_pred = output

    expected = (batch_size, num_frames, num_channels, height, width)
    print(f"Output shape: {flow_pred.shape}")
    if flow_pred.shape != expected:
        print(f"❌ Shape mismatch: expected {expected}")
        return 1

    print("✅ Train-path conditioning check passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
