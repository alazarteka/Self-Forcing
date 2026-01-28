import torch
import torch.nn.functional as F
from einops import rearrange

from wan.modules.clip import CLIPModel


class PoseImageConditioner(torch.nn.Module):
    def __init__(
        self,
        device,
        dtype,
        pose_weights_path=None,
        pose_weights_strict=True,
        clip_checkpoint_path="wan_models/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        clip_tokenizer="xlm-roberta-large",
        vae=None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.pose_weights_path = pose_weights_path
        self.pose_weights_strict = pose_weights_strict
        self.pose_weights_loaded = False

        self.dwpose_embedding = self._get_dwpose_embedding()
        self.randomref_embedding_pose = self._get_randomref_embedding_pose()
        self.dwpose_embedding.to(device)
        self.randomref_embedding_pose.to(device)

        self.image_encoder = CLIPModel(
            dtype=torch.float32,
            device=device,
            checkpoint_path=clip_checkpoint_path,
            tokenizer_path=clip_tokenizer
        )
        self.image_encoder.eval().requires_grad_(False)

        self.vae = vae
        if self.vae is None:
            raise ValueError("PoseImageConditioner requires a VAE wrapper instance.")

        self.dwpose_embedding.eval().requires_grad_(False)
        self.randomref_embedding_pose.eval().requires_grad_(False)

    def _get_dwpose_embedding(self):
        concat_dim = 4
        dwpose_embedding = torch.nn.Sequential(
            torch.nn.Conv3d(3, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            torch.nn.SiLU(),
            torch.nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            torch.nn.SiLU(),
            torch.nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            torch.nn.SiLU(),
            torch.nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.SiLU(),
            torch.nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(concat_dim * 4, 5120, (1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        return dwpose_embedding

    def _get_randomref_embedding_pose(self):
        concat_dim = 4
        randomref_dim = 20
        randomref_embedding_pose = torch.nn.Sequential(
            torch.nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
        )
        return randomref_embedding_pose

    def load_pose_embedding_weights(self):
        if self.pose_weights_loaded or self.pose_weights_path is None:
            return
        state_dict = torch.load(self.pose_weights_path, map_location="cpu")
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
            self.dwpose_embedding.load_state_dict(dwpose_sd, strict=self.pose_weights_strict)
        if randomref_sd:
            self.randomref_embedding_pose.load_state_dict(randomref_sd, strict=self.pose_weights_strict)
        if not dwpose_sd and not randomref_sd:
            raise ValueError("No pose embedding weights found in state_dict.")
        self.pose_weights_loaded = True

    @staticmethod
    def _ensure_batch(x, ndim):
        if x.ndim == ndim - 1:
            return x.unsqueeze(0)
        return x

    def encode_pose(self, dwpose_data, random_ref_dwpose):
        self.load_pose_embedding_weights()
        dwpose_data = self._ensure_batch(dwpose_data, 5).to(self.device)
        random_ref_dwpose = self._ensure_batch(random_ref_dwpose, 4).to(self.device)

        dwpose_data_emb = self.dwpose_embedding(
            (torch.cat([dwpose_data[:, :, :1].repeat(1, 1, 3, 1, 1), dwpose_data], dim=2) / 255.0).to(self.device)
        ).to(self.dtype)
        random_ref_dwpose_data = self.randomref_embedding_pose(
            (random_ref_dwpose / 255.0).to(self.device).permute(0, 3, 1, 2)
        ).unsqueeze(2).to(self.dtype)
        return dwpose_data_emb, random_ref_dwpose_data

    def encode_image(self, first_frame, num_frames, height, width):
        first_frame = self._ensure_batch(first_frame, 4).to(self.device)
        images = first_frame.float()
        if images.max() > 1:
            images = images * (2.0 / 255.0) - 1.0
        images = images.permute(0, 3, 1, 2)

        if images.shape[-2:] != (height, width):
            images = F.interpolate(images, size=(height, width), mode="bicubic", align_corners=False)

        clip_context = self.image_encoder.visual(
            [img[:, None, :, :] for img in images]
        ).to(dtype=self.dtype, device=self.device)

        lat_h = height // 8
        lat_w = width // 8
        msk = torch.ones(1, num_frames, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        y_list = []
        for img in images:
            vae_input = torch.concat(
                [img.unsqueeze(1), torch.zeros(3, num_frames - 1, height, width, device=self.device)],
                dim=1
            )
            y = self.vae.model.encode([vae_input.to(dtype=self.dtype, device=self.device)], device=self.device)[0]
            y = torch.concat([msk, y])
            y_list.append(y)
        y = torch.stack(y_list, dim=0).to(dtype=self.dtype, device=self.device)
        return clip_context, y

    def build_conditioning(
        self,
        first_frame,
        dwpose_data,
        random_ref_dwpose,
        num_frames,
        height,p
        width,
        pose_drop_prob=0.0
    ):
        clip_feature, image_y = self.encode_image(first_frame, num_frames, height, width)
        dwpose_data_emb, random_ref_dwpose_data = self.encode_pose(dwpose_data, random_ref_dwpose)
        add_condition = rearrange(dwpose_data_emb, "b c f h w -> b (f h w) c").contiguous()
        y = image_y + random_ref_dwpose_data

        if pose_drop_prob > 0.0:
            if torch.rand(1, device=self.device).item() < pose_drop_prob:
                add_condition = torch.zeros_like(add_condition)
                y = image_y

        return {
            "add_condition": add_condition,
            "clip_feature": clip_feature,
            "y": y
        }
