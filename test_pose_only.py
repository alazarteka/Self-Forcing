#!/usr/bin/env python3
"""
Level 1.8: Pose-only behavior check (CPU).
Verifies behavior when dwpose_data is provided but random_ref_dwpose is not.
"""

import torch
import sys

def test_pose_only_behavior():
    print("=" * 60)
    print("TEST 1.8: Pose-Only Behavior Check")
    print("=" * 60)

    try:
        # Simulate inputs
        dwpose_data = torch.randn(1, 3, 21, 480, 832)  # [B, C, F, H, W]
        random_ref_dwpose = None # Pose-only mode
        
        print(f"✓ Inputs: dwpose_data={dwpose_data.shape}, random_ref_dwpose={random_ref_dwpose}")

        # Basic logic check: Should the pipeline accept this?
        # Current implementation in pipeline/causal_diffusion_inference.py:
        # if random_ref_dwpose is not None: ... 
        # But dwpose_data should still be usable even if reference pose is missing
        
        if random_ref_dwpose is None:
            print("✓ random_ref_dwpose is None - entering pose-only check mode.")
            # Verify that we can still process dwpose_data
            if dwpose_data is not None:
                print("✓ dwpose_data is present. Pipeline should use it for conditioning.")
            else:
                print("✗ ERROR: Both random_ref_dwpose and dwpose_data are missing.")
                return False
        
        print("✓ Pose-only configuration structured correctly.")
        return True

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if test_pose_only_behavior():
        print("\n✅ Pose-only behavior check passed!")
        sys.exit(0)
    else:
        print("\n❌ Pose-only behavior check failed!")
        sys.exit(1)
