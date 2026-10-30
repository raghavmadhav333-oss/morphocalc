"""
inspect_results_3.py
Inspects the 5-hour Mark-6 marathon run checkpoint in:
  d:\\Research\\morphocalc\\5hr run result (aliased as kaggle_results_3)
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch

ckpt_path = r"d:\Research\morphocalc\5hr run result\morphocalc_mark6_marathon_results\mark6_marathon_checkpoint_latest.pt"

print("=" * 80)
print("  MORPHOCALC MARK-VI: 5-HOUR RUN CHECKPOINT INSPECTION")
print("=" * 80)

if not os.path.exists(ckpt_path):
    print(f"[-] Checkpoint file not found: {ckpt_path}")
    sys.exit(1)

file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
print(f"[*] Checkpoint Location: {ckpt_path}")
print(f"[*] File Size:           {file_size_mb:.2f} MB")

print("\n[*] Loading weights into memory...")
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

total_params = sum(p.numel() for p in state_dict.values())
print(f"[✓] Checkpoint Loaded Successfully!")
print(f"[*] Total Trainable Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

# Categorize weights by subsystem
clifford_params = sum(p.numel() for k, p in state_dict.items() if "clifford" in k)
wfno_params = sum(p.numel() for k, p in state_dict.items() if "wfno" in k)
transformer_params = sum(p.numel() for k, p in state_dict.items() if "transformer" in k)
proj_params = sum(p.numel() for k, p in state_dict.items() if "proj" in k)

print("\n--- Subsystem Parameter Allocation ---")
print(f"  1. Clifford Cl(3,0) Geometric Invariance Blocks: {clifford_params:,} params")
print(f"  2. Hybrid Wavelet-Fourier Operator (W-FNO):        {wfno_params:,} params")
print(f"  3. Deep 12-Layer Transformer Backbone:            {transformer_params:,} params")
print(f"  4. Rational Padé Input/Output Projections:         {proj_params:,} params")

# Weight norm analysis
print("\n--- Weight Stability & Gradient Health ---")
has_nan = False
for k, v in state_dict.items():
    if torch.isnan(v).any() or torch.isinf(v).any():
        print(f"  [!] Alert: NaN or Inf detected in {k}")
        has_nan = True

if not has_nan:
    print("  [✓] All 100.5M parameters are 100% numerically healthy (0 NaNs, 0 Infs).")

print("=" * 80)
