import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from wan.modules.model import WanSelfAttention, WanAttentionBlock
from wan.modules.causal_model import CausalWanSelfAttention, CausalWanAttentionBlock


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @classmethod
    def from_linear(
        cls,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        lora = cls(base, rank=rank, alpha=alpha, dropout=dropout)
        lora.to(device=base.weight.device, dtype=base.weight.dtype)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        out = out + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out


def _load_state_dict(path: Path):
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError as exc:
            raise ImportError("safetensors is required to load .safetensors LoRA files.") from exc
        return load_safetensors(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _find_lora_pairs(lora_sd):
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


def _target_from_lora_key(key: str) -> Optional[str]:
    parts = key.split(".")
    for token in ("lora_B", "lora_up"):
        if token in parts:
            idx = parts.index(token)
            parts.pop(idx)
            if idx < len(parts) and parts[idx] == "default":
                parts.pop(idx)
            return ".".join(parts)
    return None


def _map_target_name(name: str, base_keys: Iterable[str]) -> Optional[str]:
    if name in base_keys:
        return name
    for prefix in ("diffusion_model.", "model.", "pipe.dit.", "pipe."):
        if name.startswith(prefix):
            candidate = name[len(prefix):]
            if candidate in base_keys:
                return candidate
    return None


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Sequence[str] = ("q", "k", "v", "o"),
) -> int:
    replaced = 0
    attn_targets = [t for t in target_modules if not t.startswith("ffn.")]
    ffn_targets = []
    for target in target_modules:
        if target.startswith("ffn."):
            try:
                ffn_targets.append(int(target.split(".", 1)[1]))
            except (IndexError, ValueError):
                continue
    for module in model.modules():
        if isinstance(module, (WanSelfAttention, CausalWanSelfAttention)):
            for attr in attn_targets:
                layer = getattr(module, attr, None)
                if isinstance(layer, LoRALinear):
                    continue
                if isinstance(layer, nn.Linear):
                    setattr(module, attr, LoRALinear.from_linear(
                        layer, rank=rank, alpha=alpha, dropout=dropout
                    ))
                    replaced += 1
        if ffn_targets and isinstance(module, (WanAttentionBlock, CausalWanAttentionBlock)):
            if not isinstance(module.ffn, nn.Sequential):
                continue
            for idx in ffn_targets:
                if idx < 0 or idx >= len(module.ffn):
                    continue
                layer = module.ffn[idx]
                if isinstance(layer, LoRALinear):
                    continue
                if isinstance(layer, nn.Linear):
                    module.ffn[idx] = LoRALinear.from_linear(
                        layer, rank=rank, alpha=alpha, dropout=dropout
                    )
                    replaced += 1
    return replaced


def load_lora_weights(
    model: nn.Module,
    lora_path: str,
    alpha: Optional[float] = None,
) -> Tuple[int, int]:
    lora_sd = _load_state_dict(Path(lora_path))
    pairs = _find_lora_pairs(lora_sd)
    if not pairs:
        raise ValueError("No LoRA pairs found (expected lora_A/lora_B or lora_up/lora_down).")

    module_map = dict(model.named_modules())
    param_names = set(dict(model.named_parameters()).keys())

    loaded = 0
    skipped = 0

    for key_up, key_down in pairs.items():
        target = _target_from_lora_key(key_up)
        if target is None:
            skipped += 1
            continue

        module_target = target
        for prefix in ("diffusion_model.", "model.", "pipe.dit.", "pipe."):
            if module_target.startswith(prefix):
                module_target = module_target[len(prefix):]
                break
        if module_target.endswith(".weight") or module_target.endswith(".bias"):
            module_target = module_target.rsplit(".", 1)[0]

        module = module_map.get(module_target)
        if not isinstance(module, LoRALinear):
            mapped = _map_target_name(target, param_names)
            target = mapped or target

            candidates = [target]
            if not target.endswith((".weight", ".bias")):
                candidates.extend([f"{target}.weight", f"{target}.bias"])

            expanded = []
            for cand in candidates:
                expanded.append(cand)
                if cand.endswith(".weight"):
                    expanded.append(cand.replace(".weight", ".base.weight"))
                if cand.endswith(".bias"):
                    expanded.append(cand.replace(".bias", ".base.bias"))

            resolved = None
            for cand in expanded:
                if cand in param_names:
                    resolved = cand
                    break

            if resolved is None:
                skipped += 1
                continue

            module_name = resolved.rsplit(".", 1)[0]
            module = module_map.get(module_name)
            if isinstance(module, nn.Linear) and module_name.endswith(".base"):
                module = module_map.get(module_name[:-5], module)

            if not isinstance(module, LoRALinear):
                skipped += 1
                continue

        device = module.lora_A.weight.device
        dtype = module.lora_A.weight.dtype
        up = lora_sd[key_up].to(device=device, dtype=dtype)
        down = lora_sd[key_down].to(device=device, dtype=dtype)
        with torch.no_grad():
            module.lora_B.weight.copy_(up)
            module.lora_A.weight.copy_(down)
            if alpha is not None:
                module.alpha = alpha
                module.scaling = alpha / module.rank
        loaded += 1

    return loaded, skipped


def mark_only_lora_as_trainable(
    model: nn.Module,
    extra_trainable_keywords: Sequence[str] = (),
) -> None:
    keywords = tuple(extra_trainable_keywords)
    for name, param in model.named_parameters():
        trainable = ("lora_A" in name) or ("lora_B" in name)
        if not trainable and keywords:
            trainable = any(keyword in name for keyword in keywords)
        param.requires_grad_(trainable)
