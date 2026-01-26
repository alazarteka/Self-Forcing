from torch.fx.experimental.migrate_gradual_types.z3_types import dim
from einops import rearrange
from tqdm import tqdm
from typing import List, Optional
import torch

from wan.modules.clip import CLIPModel
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            image_encoder=None
    ):
        super().__init__()
        self.device = device
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.image_encoder = CLIPModel(
            dtype=torch.float32,
            device=device,
            checkpoint_path="wan_models/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            tokenizer_path="xlm-roberta-large"
        ) if image_encoder is None else image_encoder
        self.vae = WanVAEWrapper() if vae is None else vae
        self.dwpose_embedding = self._get_dwpose_embedding() # TODO: Implement model weight loading
        self.randomref_embedding_pose = self._get_randomref_embedding_pose()

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
    
    def _get_dwpose_embedding(self):
        CONCAT_DIM = 4
        dwpose_embedding = torch.nn.Sequential(
                    torch.nn.Conv3d(3, CONCAT_DIM * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, 3, stride=(2,2,2), padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(CONCAT_DIM * 4, CONCAT_DIM * 4, 3, stride=(2,2,2), padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(CONCAT_DIM * 4, 5120, (1,2,2), stride=(1,2,2), padding=0)
        )
        return dwpose_embedding
    
    def _get_randomref_embedding_pose(self):
        CONCAT_DIM = 4
        RANDOMREF_DIM = 20
        randomref_embedding_pose = torch.nn.Sequential(
                    torch.nn.Conv2d(3, CONCAT_DIM * 4, 3, stride=1, padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(CONCAT_DIM * 4, CONCAT_DIM * 4, 3, stride=1, padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(CONCAT_DIM * 4, CONCAT_DIM * 4, 3, stride=1, padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(CONCAT_DIM * 4, CONCAT_DIM * 4, 3, stride=2, padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(CONCAT_DIM * 4, CONCAT_DIM * 4, 3, stride=2, padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(CONCAT_DIM * 4, RANDOMREF_DIM, 3, stride=2, padding=1),
        )
        return randomref_embedding_pose

    def load_pose_embedding_weights(self, state_dict_or_path):
        if isinstance(state_dict_or_path, str):
            state_dict = torch.load(state_dict_or_path, map_location="cpu")
        else:
            state_dict = state_dict_or_path

        dwpose_sd = {
            k.split("dwpose_embedding.", 1)[1]: v
            for k, v in state_dict.items()
            if k.startswith("dwpose_embedding.")
        }
        randomref_sd = {
            k.split("randomref_embedding_pose.", 1)[1]: v
            for k, v in state_dict.items()
            if k.startswith("randomref_embedding_pose.")
        }
        if dwpose_sd:
            self.dwpose_embedding.load_state_dict(dwpose_sd, strict=True)
        if randomref_sd:
            self.randomref_embedding_pose.load_state_dict(randomref_sd, strict=True)
        if not dwpose_sd and not randomref_sd:
            raise ValueError("No pose embedding weights found in state_dict.")
    
    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0) # Normalize to [-1, 1]
        return image

    def encode_image(
        self,
        image,
        num_frames,
        height,
        width
    ):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.visual([image]) # visual is the equivalent of encode_image in this repo
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1]//4, 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.model.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}
        
    
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        input_image: Optional[torch.Tensor], # TODO: Decide if this should just be the embedding
        dwpose_data: Optional[torch.Tensor],
        random_ref_dwpose: Optional[torch.Tensor],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
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
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
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
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        
        # Below is a direct copy from UniAnimate-DiT
        # diffsynth/pipelines/wan_video.py line 707 and skipping some
        # TODO: Actually plug in the components
        # Image conditioning is not wired yet; keep placeholder dict for future use.
        if input_image is not None and self.image_encoder is not None:
            # self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}

        device = noise.device
        self.dwpose_embedding.to(device)
        self.randomref_embedding_pose.to(device)

        dwpose_data_emb = None
        if dwpose_data is not None and random_ref_dwpose is not None:
            dwpose_data = dwpose_data.unsqueeze(0)
            dwpose_data_emb = self.dwpose_embedding(
                (torch.cat([dwpose_data[:, :, :1].repeat(1, 1, 3, 1, 1), dwpose_data], dim=2) / 255.0).to(device)
            ).to(torch.bfloat16)
            random_ref_dwpose_data = self.randomref_embedding_pose(
                (random_ref_dwpose.unsqueeze(0) / 255.0).to(device).permute(0, 3, 1, 2)
            ).unsqueeze(2).to(torch.bfloat16)  # [1, 20, 104, 60]

            # TODO: integrate image_emb into the model...
            if "y" in image_emb:
                image_emb["y"] = image_emb["y"] + random_ref_dwpose_data  # image_emb is the image to be driven by the pose

        # Extract image conditioning features to pass to model
        clip_feature = image_emb.get("clip_feature", None)
        y = image_emb.get("y", None)

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                if dwpose_data_emb is not None:
                    start = current_start_frame
                    end = current_start_frame + current_num_frames
                    if end > dwpose_data_emb.shape[2]:
                        raise ValueError("dwpose_data has fewer frames than required for the current block.")
                    condition = rearrange(
                        dwpose_data_emb[:, :, start:end],
                        'b c f h w -> b (f h w) c'
                    ).contiguous()
                else:
                    condition = None

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    add_condition = condition,
                    clip_feature = clip_feature,
                    y = y
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    add_condition = None,
                    clip_feature = clip_feature,
                    y = y
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
                print(f"kv_cache['local_end_index']: {self.kv_cache_pos[0]['local_end_index']}")
                print(f"kv_cache['global_end_index']: {self.kv_cache_pos[0]['global_end_index']}")

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                add_condition = condition,
                clip_feature = clip_feature,
                y = y
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                add_condition = None,
                clip_feature = clip_feature,
                y = y
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_pos = []
        kv_cache_neg = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
