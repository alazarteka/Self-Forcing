#!/usr/bin/env python3
"""
Level 1.5: Pose-frame alignment smoke check (CPU).
Asserts that pose frames match the output timeline.
"""

import torch
import sys

def test_pose_frame_alignment():
    print("=" * 60)
    print("TEST 1.5: Pose-Frame Alignment Smoke Check")
    print("=" * 60)

    try:
        # Configuration
        num_frames = 21
        num_frame_per_block = 3
        
        # Simulate UniAnimate-style pose embedding
        # Input dwpose_data: [B, 3, 81, H, W] -> prepend 3 frames -> 84
        # After dwpose_embedding, expect 21 frames
        height = 64
        width = 48
        dwpose_data = torch.randn(1, 3, 81, height, width)
        dwpose_input = torch.cat(
            [dwpose_data[:, :, :1].repeat(1, 1, 3, 1, 1), dwpose_data], dim=2
        )

        # Create the same embedding stack used in the pipeline
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

        dwpose_data_emb = dwpose_embedding(dwpose_input)
        print(f"✓ Created dummy pose embedding: {dwpose_data_emb.shape}")
        if dwpose_data_emb.shape[2] != num_frames:
            print(f"✗ ERROR: Expected {num_frames} frames, got {dwpose_data_emb.shape[2]}")
            return False

        # Verification logic (as in pipeline)
        total_blocks = num_frames // num_frame_per_block
        print(f"✓ Total blocks to process: {total_blocks}")

        for i in range(total_blocks):
            start = i * num_frame_per_block
            end = (i + 1) * num_frame_per_block
            
            # Assert alignment
            if end > dwpose_data_emb.shape[2]:
                print(f"✗ ERROR: Block [{start}:{end}] exceeds pose frames ({dwpose_data_emb.shape[2]})")
                return False
            
            # Simulate extraction
            block_pose = dwpose_data_emb[:, :, start:end]
            if block_pose.shape[2] != num_frame_per_block:
                print(f"✗ ERROR: Extracted block size {block_pose.shape[2]} != {num_frame_per_block}")
                return False
                
        print("✓ All blocks aligned with pose data frames.")
        return True

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if test_pose_frame_alignment():
        print("\n✅ Pose alignment check passed!")
        sys.exit(0)
    else:
        print("\n❌ Pose alignment check failed!")
        sys.exit(1)
