import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from utils.wan_wrapper import WanDiffusionWrapper


def _make_kv_cache(batch_size: int, cache_size: int, num_heads: int, head_dim: int, device, dtype):
    return {
        "k": torch.zeros([batch_size, cache_size, num_heads, head_dim], device=device, dtype=dtype),
        "v": torch.zeros([batch_size, cache_size, num_heads, head_dim], device=device, dtype=dtype),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
    }


def _make_crossattn_cache(batch_size: int, text_len: int, num_heads: int, head_dim: int, device, dtype):
    return {
        "k": torch.zeros([batch_size, text_len, num_heads, head_dim], device=device, dtype=dtype),
        "v": torch.zeros([batch_size, text_len, num_heads, head_dim], device=device, dtype=dtype),
        "is_init": False,
    }


def _make_kv_cache_list(num_blocks: int, *args, **kwargs):
    return [_make_kv_cache(*args, **kwargs) for _ in range(num_blocks)]


def _make_crossattn_cache_list(num_blocks: int, *args, **kwargs):
    return [_make_crossattn_cache(*args, **kwargs) for _ in range(num_blocks)]


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32

    model = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        is_causal=True,
        local_attn_size=1,
        use_vace=True,
    ).to(device=device, dtype=dtype)

    batch_size = 1
    num_frames = 1
    c_in = 16
    height = 8
    width = 8

    noisy = torch.randn([batch_size, num_frames, c_in, height, width], device=device, dtype=dtype)
    vace_context = torch.randn([batch_size, num_frames, c_in, height, width], device=device, dtype=dtype)

    prompt_embeds = torch.randn([batch_size, 512, 4096], device=device, dtype=dtype)
    conditional_dict = {"prompt_embeds": prompt_embeds}

    timestep = torch.zeros([batch_size, num_frames], device=device, dtype=torch.float32)

    num_layers = model.model.num_layers
    num_heads = model.model.num_heads
    head_dim = model.model.dim // model.model.num_heads
    frame_seqlen = (height // model.model.patch_size[1]) * (width // model.model.patch_size[2])
    kv_cache_size = frame_seqlen

    kv_cache = _make_kv_cache_list(num_layers, batch_size, kv_cache_size, num_heads, head_dim, device, dtype)
    crossattn_cache = _make_crossattn_cache_list(num_layers, batch_size, 512, num_heads, head_dim, device, dtype)

    vace_num_blocks = len(model.model.vace_blocks)
    vace_kv_cache = _make_kv_cache_list(
        vace_num_blocks, batch_size, kv_cache_size, num_heads, head_dim, device, dtype
    )
    vace_crossattn_cache = _make_crossattn_cache_list(
        vace_num_blocks, batch_size, 512, num_heads, head_dim, device, dtype
    )

    with torch.no_grad():
        flow_pred, pred_x0 = model(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=0,
            cache_start=0,
            vace_context=vace_context,
            vace_context_scale=1.0,
            vace_kv_cache=vace_kv_cache,
            vace_crossattn_cache=vace_crossattn_cache,
        )

    print("flow_pred", tuple(flow_pred.shape), flow_pred.dtype, flow_pred.device)
    print("pred_x0", tuple(pred_x0.shape), pred_x0.dtype, pred_x0.device)


if __name__ == "__main__":
    main()
