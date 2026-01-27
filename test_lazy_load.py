import torch
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from argparse import Namespace

def test_lazy_load():
    print("============================================================")
    print("LAZY POSE-WEIGHT LOAD CHECK")
    print("============================================================")

    # 1. Create a dummy pose checkpoint
    checkpoint_path = "dummy_pose_weights.pt"
    print(f"Creating dummy checkpoint at {checkpoint_path}...")
    
    # We need to match the structure expected by load_pose_embedding_weights
    # dwpose_embedding: Sequential with various Conv3d layers
    # randomref_embedding_pose: Sequential with various Conv2d layers
    
    dummy_sd = {
        "dwpose_embedding.0.weight": torch.randn(16, 3, 3, 3, 3),
        "randomref_embedding_pose.0.weight": torch.randn(16, 3, 3, 3)
    }
    torch.save(dummy_sd, checkpoint_path)

    # 2. Initialize pipeline with lazy config
    print("Initializing pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Namespace(
        model_kwargs={"is_causal": True},
        num_train_timestep=1000,
        timestep_shift=1.0,
        independent_first_frame=True,
        pose_weights_path=checkpoint_path,
        pose_weights_strict=False, # Use False so it doesn't complain about missing layers
        negative_prompt=""
    )
    
    # We mock out the heavy models to focus on the pipeline logic
    mock_generator = torch.nn.Module()
    mock_generator.model = torch.nn.Module()
    mock_generator.model.local_attn_size = -1
    
    pipeline = CausalDiffusionInferencePipeline(args, device, 
                                               generator=mock_generator,
                                               text_encoder=lambda text_prompts: {"prompt_embeds": torch.randn(1, 512, 4096)},
                                               vae=torch.nn.Module(), # Mock
                                               image_encoder=torch.nn.Module()) # Mock

    print(f"Initial pose_weights_loaded: {pipeline.pose_weights_loaded}")
    
    # 3. Simulate part of the inference loop to trigger load
    # We don't need to run the whole thing, just enough to hit the loading block
    print("Simulating inference trigger...")
    
    # Mocking noisy input
    noise = torch.randn(1, 4, 16, 8, 8).to(device)
    
    try:
        # This will fail eventually because of our mocks, but we just want to see if it loads
        pipeline.inference(
            noise=noise,
            text_prompts=["test"],
            input_image=None,
            dwpose_data=None, # Set to None to avoid deeper logic but trigger the load block
            random_ref_dwpose=None
        )
    except Exception as e:
        print(f"Inference interrupted (as expected): {type(e).__name__}")

    print(f"Final pose_weights_loaded: {pipeline.pose_weights_loaded}")
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    if pipeline.pose_weights_loaded:
        print("\n✅ LAZY LOAD VERIFIED: weights were loaded on demand.")
        return True
    else:
        print("\n❌ LAZY LOAD FAILED: weights were not loaded.")
        return False

if __name__ == "__main__":
    if not test_lazy_load():
        sys.exit(1)
