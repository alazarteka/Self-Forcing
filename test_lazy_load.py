import argparse
import os
import sys
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from argparse import Namespace


def build_args(pose_weights_path, pose_weights_strict):
    return Namespace(
        model_kwargs={"is_causal": True},
        num_train_timestep=1000,
        timestep_shift=1.0,
        independent_first_frame=True,
        pose_weights_path=pose_weights_path,
        pose_weights_strict=pose_weights_strict,
        negative_prompt=""
    )


def create_dummy_checkpoint(checkpoint_path):
    print(f"Creating dummy checkpoint at {checkpoint_path}...")
    
    # We need to provide all keys if we want to support --strict
    # Instead of hardcoding, we can initialize temporary modules and get their state dicts
    from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
    
    # Mock namespace for initialization
    class MockArgs:
        num_train_timestep = 1000
    
    # Temporarily create a pipeline just to get the state dict structure
    # We use a very minimal mock to avoid errors
    mock_gen = torch.nn.Module()
    mock_gen.model = torch.nn.Module()
    mock_gen.model.local_attn_size = -1
    
    pipeline = CausalDiffusionInferencePipeline(
        Namespace(model_kwargs={}, num_train_timestep=1000, timestep_shift=1.0, independent_first_frame=True),
        torch.device("cpu"),
        generator=mock_gen,
        text_encoder=lambda x: {},
        vae=torch.nn.Module(),
        image_encoder=torch.nn.Module()
    )
    
    dummy_sd = {}
    for k, v in pipeline.dwpose_embedding.state_dict().items():
        dummy_sd[f"dwpose_embedding.{k}"] = torch.randn_like(v)
    for k, v in pipeline.randomref_embedding_pose.state_dict().items():
        dummy_sd[f"randomref_embedding_pose.{k}"] = torch.randn_like(v)
        
    torch.save(dummy_sd, checkpoint_path)
    print(f"Created complete dummy checkpoint with {len(dummy_sd)} keys.")


def test_lazy_load(pose_weights_path=None, pose_weights_strict=None):
    print("============================================================")
    print("LAZY POSE-WEIGHT LOAD CHECK")
    print("============================================================")

    created_dummy = False
    if pose_weights_path is None:
        checkpoint_path = "dummy_pose_weights.pt"
        create_dummy_checkpoint(checkpoint_path)
        created_dummy = True
        if pose_weights_strict is None:
            pose_weights_strict = False
    else:
        checkpoint_path = pose_weights_path
        if pose_weights_strict is None:
            pose_weights_strict = True

    # 2. Initialize pipeline with lazy config
    print("Initializing pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = build_args(checkpoint_path, pose_weights_strict)
    
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
        import traceback
        traceback.print_exc()
        print(f"Inference interrupted (as expected): {type(e).__name__}")

    print(f"Final pose_weights_loaded: {pipeline.pose_weights_loaded}")
    
    # Cleanup
    if created_dummy and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    if pipeline.pose_weights_loaded:
        print("\n✅ LAZY LOAD VERIFIED: weights were loaded on demand.")
        return True
    else:
        print("\n❌ LAZY LOAD FAILED: weights were not loaded.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lazy pose-weight load test")
    parser.add_argument("--pose-weights-path", dest="pose_weights_path", default=None)
    parser.add_argument("--strict", dest="pose_weights_strict", action="store_true")
    parser.add_argument("--no-strict", dest="pose_weights_strict", action="store_false")
    parser.set_defaults(pose_weights_strict=None)
    args = parser.parse_args()

    if not test_lazy_load(
        pose_weights_path=args.pose_weights_path,
        pose_weights_strict=args.pose_weights_strict
    ):
        sys.exit(1)
