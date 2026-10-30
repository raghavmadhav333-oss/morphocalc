import os
import torch

ckpt_path = r"d:\Research\morphocalc\overnight run\morphocalc_nexus_grand_overnight.pt"
print("Loading checkpoint:", ckpt_path)
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)

print("Total items:", len(state))
tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
non_tensors = {k: v for k, v in state.items() if not isinstance(v, torch.Tensor)}

print(f"Tensor parameters count: {len(tensors)}")
print(f"Non-tensor metadata keys: {list(non_tensors.keys())}")

total_params = sum(p.numel() for p in tensors.values())
print(f"Total Parameter Count: {total_params:,}")

# Analyze model architecture layers
layer_prefixes = set()
for k in tensors.keys():
    prefix = k.split('.')[0] if '.' in k else k
    layer_prefixes.add(prefix)

print("Architecture Modules:", sorted(list(layer_prefixes)))

# Inspect individual transformer blocks
blocks = set()
for k in tensors.keys():
    if "block" in k or "layer" in k or "encoder" in k:
        parts = k.split('.')
        for p in parts:
            if any(p.startswith(x) for x in ["block", "layer", "encoder", "transformer"]):
                blocks.add(p)
print("Detected Transformer Sub-Blocks:", sorted(list(blocks))[:15])

# Print first 20 tensor shapes
print("\nFirst 20 Tensors:")
for i, (k, v) in enumerate(list(tensors.items())[:20]):
    print(f"  [{i:2d}] {k:<45} : shape={list(v.shape)}, dtype={v.dtype}, mean={v.float().mean().item():.4e}, std={v.float().std().item():.4e}")

# Check for SINDY or physical closure coefficients
sindy_keys = [k for k in tensors.keys() if any(x in k.lower() for x in ["sindy", "coeff", "pope", "closure", "poly", "rational"])]
print("\nPhysical / SINDY Closure Keys:", sindy_keys)
for k in sindy_keys:
    print(f"  {k}: {tensors[k]}")
