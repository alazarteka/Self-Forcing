#!/usr/bin/env python3
"""
Offline LoRA merge utility for Wan2.1 checkpoints.

This merges LoRA weights into a base Wan checkpoint and saves a merged state_dict.
It is intended for teacher preparation (e.g., UniAnimate 14B + LoRA) before distillation.
"""

import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


def load_state_dict(path: Path):
    if path.suffix == ".safetensors":
        return load_safetensors(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


def save_state_dict(state_dict, path: Path):
    if path.suffix == ".safetensors":
        save_safetensors(state_dict, str(path))
    else:
        torch.save(state_dict, str(path))


def find_lora_pairs(lora_sd):
    pairs = {}
    for key in lora_sd.keys():
        if "lora_B" in key:
            key_a = key.replace("lora_B", "lora_A")
            if key_a in lora_sd:
                pairs[key] = key_a
        elif "lora_up" in key:
            key_down = key.replace("lora_up", "lora_down")
            if key_down in lora_sd:
                pairs[key] = key_down
    return pairs


def target_from_lora_key(key):
    parts = key.split(".")
    for token in ("lora_B", "lora_up"):
        if token in parts:
            idx = parts.index(token)
            parts.pop(idx)
            if idx < len(parts) and parts[idx] == "default":
                parts.pop(idx)
            return ".".join(parts)
    return None


def map_target_name(name, base_keys):
    if name in base_keys:
        return name
    for prefix in ("diffusion_model.", "model.", "pipe.dit.", "pipe."):
        if name.startswith(prefix):
            candidate = name[len(prefix):]
            if candidate in base_keys:
                return candidate
    return None


def apply_lora_to_state_dict(base_sd, lora_sd, alpha=1.0, dtype=None, device="cpu"):
    base_keys = set(base_sd.keys())
    pairs = find_lora_pairs(lora_sd)
    if not pairs:
        raise ValueError("No LoRA pairs found (expected lora_A/lora_B or lora_up/lora_down).")

    updated = 0
    skipped = 0

    for key_up, key_down in pairs.items():
        target = target_from_lora_key(key_up)
        if target is None:
            skipped += 1
            continue

        target = map_target_name(target, base_keys)
        if target is None:
            skipped += 1
            continue

        weight_up = lora_sd[key_up]
        weight_down = lora_sd[key_down]

        # Move to compute dtype/device
        weight_up = weight_up.to(device=device, dtype=dtype or weight_up.dtype)
        weight_down = weight_down.to(device=device, dtype=dtype or weight_down.dtype)

        # Compute delta
        if weight_up.ndim == 4:
            weight_up = weight_up.squeeze(3).squeeze(2)
            weight_down = weight_down.squeeze(3).squeeze(2)
            delta = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            delta = alpha * torch.mm(weight_up, weight_down)

        base_weight = base_sd[target]
        base_dtype = base_weight.dtype
        base_device = base_weight.device
        base_weight = base_weight.to(device=delta.device, dtype=delta.dtype)

        base_weight = base_weight + delta
        base_sd[target] = base_weight.to(device=base_device, dtype=base_dtype)
        updated += 1

    return updated, skipped


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into Wan2.1 base checkpoint")
    parser.add_argument("--base", required=True, help="Path to base checkpoint (.pt or .safetensors)")
    parser.add_argument("--lora", required=True, help="Path to LoRA checkpoint (.pt or .safetensors)")
    parser.add_argument("--out", required=True, help="Output path for merged checkpoint")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha (scale)")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32", "bfloat16"], help="Compute dtype")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu or cuda:0)")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    compute_dtype = dtype_map[args.dtype]

    base_path = Path(args.base)
    lora_path = Path(args.lora)
    out_path = Path(args.out)

    print(f"Loading base checkpoint: {base_path}")
    base_sd = load_state_dict(base_path)
    print(f"  Base keys: {len(base_sd)}")

    print(f"Loading LoRA checkpoint: {lora_path}")
    lora_sd = load_state_dict(lora_path)
    print(f"  LoRA keys: {len(lora_sd)}")

    print(f"Applying LoRA (alpha={args.alpha}) on device={args.device}, dtype={args.dtype} ...")
    updated, skipped = apply_lora_to_state_dict(
        base_sd, lora_sd, alpha=args.alpha, dtype=compute_dtype, device=args.device
    )

    print(f"Updated {updated} tensors, skipped {skipped}.")
    print(f"Saving merged checkpoint: {out_path}")
    save_state_dict(base_sd, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
