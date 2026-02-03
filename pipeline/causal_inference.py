from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = getattr(self.generator.model, "num_layers", 30)
        self.frame_seq_length = None

        self.kv_cache1 = None
        self.vace_kv_cache = None
        self.vace_crossattn_cache = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        vace_context: Optional[torch.Tensor] = None,
        vace_context_scale: float = 1.0,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        frame_seq_length = (height // self.generator.model.patch_size[1]) * (width // self.generator.model.patch_size[2])
        if self.frame_seq_length is None:
            self.frame_seq_length = frame_seq_length
        elif self.frame_seq_length != frame_seq_length:
            raise ValueError(
                f"frame_seq_length mismatch: cached={self.frame_seq_length}, got={frame_seq_length}. "
                "This pipeline assumes a fixed latent resolution per instance."
            )

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        vace_context_full = None
        if vace_context is not None:
            if not getattr(self.generator, "use_vace", False):
                raise ValueError("vace_context was provided but the generator was not initialized with use_vace=True.")
            if not isinstance(vace_context, torch.Tensor) or vace_context.ndim != 5:
                raise TypeError("vace_context must be a torch.Tensor with shape [B, F, C, H, W].")
            if vace_context.shape[0] != batch_size:
                raise ValueError(f"vace_context batch size mismatch: expected {batch_size}, got {vace_context.shape[0]}.")
            if vace_context.shape[-2:] != (height, width):
                raise ValueError(
                    f"vace_context spatial size mismatch: expected {(height, width)}, got {tuple(vace_context.shape[-2:])}."
                )
            if vace_context.device != noise.device or vace_context.dtype != noise.dtype:
                vace_context = vace_context.to(device=noise.device, dtype=noise.dtype)

            if vace_context.shape[1] == num_output_frames:
                vace_context_full = vace_context
            elif vace_context.shape[1] == num_frames:
                if num_input_frames > 0:
                    pad = torch.zeros(
                        [batch_size, num_input_frames, vace_context.shape[2], height, width],
                        device=noise.device,
                        dtype=noise.dtype,
                    )
                    vace_context_full = torch.cat([pad, vace_context], dim=1)
                else:
                    vace_context_full = vace_context
            else:
                raise ValueError(
                    f"vace_context has {vace_context.shape[1]} frames, but expected {num_frames} (noise) or "
                    f"{num_output_frames} (output incl. initial_latent)."
                )

            expected_vace_in_dim = getattr(self.generator.model, "vace_in_dim", None)
            if expected_vace_in_dim is not None and vace_context_full.shape[2] != expected_vace_in_dim:
                raise ValueError(
                    f"vace_context has C={vace_context_full.shape[2]}, but model expects vace_in_dim={expected_vace_in_dim}. "
                    "Set args.model_kwargs.vace_in_dim to match your VACE context."
                )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(len(self.crossattn_cache)):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        if vace_context_full is not None:
            if self.vace_kv_cache is None or self.vace_crossattn_cache is None:
                self._initialize_vace_kv_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device,
                )
                self._initialize_vace_crossattn_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device,
                )
            else:
                for block_index in range(len(self.vace_crossattn_cache)):
                    self.vace_crossattn_cache[block_index]["is_init"] = False
                for block_index in range(len(self.vace_kv_cache)):
                    self.vace_kv_cache[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.vace_kv_cache[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                gen_kwargs = dict(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=torch.zeros([batch_size, 1], device=noise.device, dtype=torch.int64),
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                if vace_context_full is not None:
                    gen_kwargs.update(
                        dict(
                            vace_context=vace_context_full[:, current_start_frame: current_start_frame + 1],
                            vace_context_scale=vace_context_scale,
                            vace_kv_cache=self.vace_kv_cache,
                            vace_crossattn_cache=self.vace_crossattn_cache,
                        )
                    )
                self.generator(**gen_kwargs)
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                gen_kwargs = dict(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=torch.zeros([batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64),
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                if vace_context_full is not None:
                    gen_kwargs.update(
                        dict(
                            vace_context=vace_context_full[
                                :, current_start_frame: current_start_frame + self.num_frame_per_block
                            ],
                            vace_context_scale=vace_context_scale,
                            vace_kv_cache=self.vace_kv_cache,
                            vace_crossattn_cache=self.vace_crossattn_cache,
                        )
                    )
                self.generator(**gen_kwargs)
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    gen_kwargs = dict(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    if vace_context_full is not None:
                        gen_kwargs.update(
                            dict(
                                vace_context=vace_context_full[
                                    :, current_start_frame: current_start_frame + current_num_frames
                                ],
                                vace_context_scale=vace_context_scale,
                                vace_kv_cache=self.vace_kv_cache,
                                vace_crossattn_cache=self.vace_crossattn_cache,
                            )
                        )
                    _, denoised_pred = self.generator(**gen_kwargs)
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    gen_kwargs = dict(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    if vace_context_full is not None:
                        gen_kwargs.update(
                            dict(
                                vace_context=vace_context_full[
                                    :, current_start_frame: current_start_frame + current_num_frames
                                ],
                                vace_context_scale=vace_context_scale,
                                vace_kv_cache=self.vace_kv_cache,
                                vace_crossattn_cache=self.vace_crossattn_cache,
                            )
                        )
                    _, denoised_pred = self.generator(**gen_kwargs)

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            gen_kwargs = dict(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            if vace_context_full is not None:
                gen_kwargs.update(
                    dict(
                        vace_context=vace_context_full[:, current_start_frame: current_start_frame + current_num_frames],
                        vace_context_scale=vace_context_scale,
                        vace_kv_cache=self.vace_kv_cache,
                        vace_crossattn_cache=self.vace_crossattn_cache,
                    )
                )
            self.generator(**gen_kwargs)

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads
        text_len = getattr(self.generator.model, "text_len", 512)
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _initialize_vace_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the VACE blocks (causal path).
        """
        num_vace_blocks = len(getattr(self.generator.model, "vace_blocks", []))
        if num_vace_blocks == 0:
            raise ValueError("use_vace=True but the model has no vace_blocks.")

        vace_kv_cache = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760

        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads
        for _ in range(num_vace_blocks):
            vace_kv_cache.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.vace_kv_cache = vace_kv_cache

    def _initialize_vace_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the VACE blocks (causal path).
        """
        num_vace_blocks = len(getattr(self.generator.model, "vace_blocks", []))
        if num_vace_blocks == 0:
            raise ValueError("use_vace=True but the model has no vace_blocks.")

        vace_crossattn_cache = []
        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads
        text_len = getattr(self.generator.model, "text_len", 512)
        for _ in range(num_vace_blocks):
            vace_crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                    "is_init": False,
                }
            )

        self.vace_crossattn_cache = vace_crossattn_cache
