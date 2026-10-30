import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import os

ckpt_path = r"d:\Research\morphocalc\kaggle_results_2\morphocalc_nexus_3d_aerospace_finetuned.pt"
print(f"Loading checkpoint: {ckpt_path}")
print(f"File size: {os.path.getsize(ckpt_path) / (1024*1024):.2f} MB")

data = torch.load(ckpt_path, map_location='cpu')

if isinstance(data, dict):
    print(f"Dict keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} entries (e.g. state_dict)")
            # Print sample weights
            sample_keys = list(v.keys())[:5]
            for sk in sample_keys:
                print(f"    - {sk}: shape={v[sk].shape}")
        else:
            print(f"  {k}: {v}")
else:
    print(f"Data type: {type(data)}")
