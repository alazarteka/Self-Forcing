from huggingface_hub import snapshot_download, hf_hub_download
import os

# Level 3: Wan2.1 weights
print("Downloading Wan2.1-T2V-1.3B...")
snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    local_dir="wan_models/Wan2.1-T2V-1.3B",
    local_dir_use_symlinks=False
)

# Level 4: Self-Forcing checkpoint
print("Downloading Self-Forcing checkpoint...")
hf_hub_download(
    repo_id="gdhe17/Self-Forcing",
    filename="checkpoints/self_forcing_dmd.pt",
    local_dir="."
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
    local_dir="."
)
