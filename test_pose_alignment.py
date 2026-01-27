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
        
        # Simulate dwpose_data_emb from pipeline
        # [B, C, F, H, W]
        dwpose_data_emb = torch.randn(1, 5120, num_frames, 60, 104)
        print(f"✓ Created dummy pose embedding: {dwpose_data_emb.shape}")

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
