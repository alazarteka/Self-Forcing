# Pose Distillation Paths

This document outlines two viable training paths to obtain a **pose‑conditioned 1.3B Self‑Forcing model** with **4‑step generation** using a UniAnimate‑style 14B teacher.

## Path A — One‑Stage Pose‑Conditioned DMD/SiD

**Goal:** Combine pose conditioning and 4‑step distillation in a single stage.

**Steps**
1. **ODE init (causal)**  
   Initialize Wan2.1‑1.3B with causal masking using ODE pairs (as in Self‑Forcing).
2. **Pose‑conditioned DMD/SiD (4 steps)**  
   - Teacher: Wan2.1‑14B + LoRA merged offline + pose encoders.  
   - Student: Wan2.1‑1.3B causal (Self‑Forcing).  
   - Conditioning: prompt + `dwpose_data` + `random_ref_dwpose`.

**Pros**
- Fastest path (single distillation stage).
- Directly optimizes for 4‑step pose‑conditioned inference.

**Risks**
- Student may be weak at pose control early in training.  
  If instability appears, switch to Path B.

---

## Path B — Two‑Stage (Pose Finetune → DMD/SiD)

**Goal:** Stabilize pose conditioning before 4‑step distillation.

**Steps**
1. **ODE init (causal)**  
   Same as Path A.
2. **Pose‑conditioned finetune (diffusion or GAN)**  
   - Train the 1.3B causal student to follow pose conditioning at standard step counts.  
   - Objective: establish strong pose adherence.
3. **Pose‑conditioned DMD/SiD (4 steps)**  
   Distill to fast generation using the 14B teacher.

**Pros**
- More stable and robust pose control.
- Lower risk of “pose‑ignorant” student.

**Risks**
- Additional training time and compute.

---

## Common Requirements

- **Teacher preparation:** offline LoRA merge before distillation.  
  See `scripts/merge_lora.py`.
- **Pose alignment:** ensure pose embedding frames match latent frames.  
  The inference pipeline enforces this with an assertion.
- **Conditioning inputs:** `dwpose_data` and `random_ref_dwpose` must be available during training.

## Recommendation

Start with **Path A** for speed. If pose adherence is unstable, fall back to **Path B** and add the intermediate finetune stage.
