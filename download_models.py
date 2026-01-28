from huggingface_hub import snapshot_download, hf_hub_download
import os

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Level 3: Wan2.1 weights
# NOTE: Commented out since we plan to start from the ODE-init Self-Forcing checkpoint.
# print("Downloading Wan2.1-T2V-1.3B...")
# snapshot_download(
#     repo_id="Wan-AI/Wan2.1-T2V-1.3B",
#     local_dir="wan_models/Wan2.1-T2V-1.3B",
#     local_dir_use_symlinks=False
# )

# Level 4: Self-Forcing checkpoint
print("Downloading Self-Forcing checkpoint...")
hf_hub_download(
    repo_id="gdhe17/Self-Forcing",
    filename="checkpoints/self_forcing_dmd.pt",
    local_dir=CHECKPOINT_DIR
)

# CLIP weights (from I2V repo)
print("Downloading CLIP weights (from Wan2.1-I2V-14B-480P)...")
hf_hub_download(
    repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
    filename="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    local_dir="wan_models/Wan2.1-T2V-1.3B"
)

# Also ODE init for good measure
print("Downloading ODE init checkpoint...")
hf_hub_download(
    repo_id="gdhe17/Self-Forcing",
    filename="checkpoints/ode_init.pt",
    local_dir=CHECKPOINT_DIR
)

# Phase 2: UniAnimate teacher assets (LoRA + pose modules)
print("Downloading Wan2.1-I2V-14B-720P base...")
snapshot_download(
    repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
    local_dir=os.path.join(CHECKPOINT_DIR, "Wan2.1-I2V-14B-720P"),
    local_dir_use_symlinks=False
)

print("Downloading UniAnimate-DiT checkpoints (LoRA + pose modules)...")
# TODO: verify downloads and load paths with a real UniAnimate teacher run.
hf_hub_download(
    repo_id="ZheWang123/UniAnimate-DiT",
    filename="UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
    local_dir=CHECKPOINT_DIR
)
hf_hub_download(
    repo_id="ZheWang123/UniAnimate-DiT",
    filename="dw-ll_ucoco_384.onnx",
    local_dir=CHECKPOINT_DIR
)
hf_hub_download(
    repo_id="ZheWang123/UniAnimate-DiT",
    filename="yolox_l.onnx",
    local_dir=CHECKPOINT_DIR
)
