import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from wan.modules.causal_model import CausalWanAttentionBlock, CausalWanModel
from wan.modules.model import sinusoidal_embedding_1d


class CausalVaceWanAttentionBlock(CausalWanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=0,
    ):
        super().__init__(
            cross_attn_type=cross_attn_type,
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.block_id = block_id

        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class CausalVaceWanModel(CausalWanModel):
    @register_to_config
    def __init__(
        self,
        vace_layers=None,
        vace_in_dim=None,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )

        self.vace_layers = (
            [i for i in range(0, self.num_layers, 2)] if vace_layers is None else list(vace_layers)
        )
        self.vace_in_dim = self.in_dim if vace_in_dim is None else int(vace_in_dim)

        if 0 not in self.vace_layers:
            raise ValueError("VACE expects layer 0 to be in vace_layers.")
        if any((i < 0 or i >= self.num_layers) for i in self.vace_layers):
            raise ValueError("All vace_layers must be within [0, num_layers).")

        self.vace_layers_mapping = {layer_idx: vace_idx for vace_idx, layer_idx in enumerate(self.vace_layers)}

        cross_attn_type = "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        self.vace_blocks = nn.ModuleList(
            [
                CausalVaceWanAttentionBlock(
                    cross_attn_type=cross_attn_type,
                    dim=self.dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    local_attn_size=self.local_attn_size,
                    sink_size=sink_size,
                    qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm,
                    eps=self.eps,
                    block_id=i,
                )
                for i in range(len(self.vace_layers))
            ]
        )

        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def _forward_vace(
        self,
        x_tokens,
        vace_context,
        kwargs,
        vace_kv_cache,
        vace_crossattn_cache,
        current_start,
        cache_start,
    ):
        if vace_kv_cache is None or vace_crossattn_cache is None:
            raise ValueError("vace_kv_cache and vace_crossattn_cache are required when vace_context is provided.")
        if len(vace_kv_cache) != len(self.vace_blocks) or len(vace_crossattn_cache) != len(self.vace_blocks):
            raise ValueError("vace_kv_cache/vace_crossattn_cache must have length == len(vace_blocks).")

        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat(c, dim=0)

        vace_kwargs = dict(x=x_tokens)
        vace_kwargs.update(kwargs)

        for block_index, block in enumerate(self.vace_blocks):
            vace_kwargs.update(
                {
                    "kv_cache": vace_kv_cache[block_index],
                    "crossattn_cache": vace_crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                }
            )
            c = block(c, **vace_kwargs)

        return torch.unbind(c)[:-1]

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start: int = 0,
        cache_start: int = 0,
        vace_context=None,
        vace_context_scale: float = 1.0,
        vace_kv_cache=None,
        vace_crossattn_cache=None,
    ):
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x, dim=0)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
        )

        hints = None
        if vace_context is not None:
            hints = self._forward_vace(
                x_tokens=x,
                vace_context=vace_context,
                kwargs=kwargs,
                vace_kv_cache=vace_kv_cache,
                vace_crossattn_cache=vace_crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
            )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)

            if hints is not None and block_index in self.vace_layers_mapping:
                x = x + hints[self.vace_layers_mapping[block_index]] * vace_context_scale

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        kwargs.pop("vace_context", None)
        kwargs.pop("vace_context_scale", None)
        kwargs.pop("vace_kv_cache", None)
        kwargs.pop("vace_crossattn_cache", None)
        return super().forward(*args, **kwargs)
