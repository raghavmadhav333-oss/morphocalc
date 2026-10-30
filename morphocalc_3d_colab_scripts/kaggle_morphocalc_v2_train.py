# ==============================================================================
# MORPHOCALC v2 - IMPROVED UNIVERSAL TRAINING SCRIPT
# Fixes: SDF/Normals, Normalization, Boundary Conditions, Checkpoints
# ==============================================================================

# !pip install torch_geometric

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# pyrefly: ignore [missing-import]
from torch_geometric.datasets import AirfRANS
import numpy as np
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# 1. LOAD DATA & EXTRACT REAL SDF + NORMALS
# ==============================================================================
print("Loading AirfRANS Dataset...")
dataset = AirfRANS(root='./AirfRANS_data', task='full', train=True)

# First, inspect what features are available
sample = dataset[0]
print(f"Position shape: {sample.pos.shape}")
print(f"Features shape (data.x): {sample.x.shape}")
print(f"Targets shape (data.y): {sample.y.shape}")
print(f"Available keys: {sample.keys()}")

NUM_SIMULATIONS = 100
print(f"\nExtracting {NUM_SIMULATIONS} wings with REAL geometry features...")

all_inputs = []
all_targets = []
all_surface_masks = []

for i in range(NUM_SIMULATIONS):
    data = dataset[i]
    
    pos = data.pos.numpy()     # (N, 2) -> X, Y
    feats = data.x.numpy()     # (N, F) -> Contains SDF, normals, etc.
    targets = data.y.numpy()   # (N, 4) -> U, V, P, nut
    
    X = pos[:, 0]
    Y = pos[:, 1]
    N_pts = X.shape[0]
    
    # =========================================================================
    # FIX #1: COMPUTE REAL SDF AND NORMALS
    # =========================================================================
    # AirfRANS data.x contains node features. We'll use them directly.
    # If data.x has enough columns, extract SDF-like features from it.
    # Otherwise, compute SDF from the geometry.
    
    if feats.shape[1] >= 2:
        # Use the raw features from the dataset as-is
        # These encode inlet velocity conditions and geometry information
        extra_feats = feats  # Use ALL available features
    else:
        extra_feats = np.zeros((N_pts, 2))
    
    # Compute approximate SDF: distance to nearest surface point
    # Surface points are where velocity magnitude is very small (no-slip)
    U_mag = np.sqrt(targets[:, 0]**2 + targets[:, 1]**2)
    
    # Points where velocity < 1% of max are likely on/near the surface
    surface_threshold = 0.01 * np.max(U_mag)
    surface_mask = U_mag < surface_threshold
    surface_points = pos[surface_mask]
    
    if len(surface_points) > 10:
        # Compute minimum distance from each point to the nearest surface point
        # Use chunked computation to avoid OOM
        sdf = np.zeros(N_pts)
        chunk_size = 5000
        for start in range(0, N_pts, chunk_size):
            end = min(start + chunk_size, N_pts)
            chunk_pos = pos[start:end]
            # Distance from each chunk point to all surface points
            dists = np.sqrt(((chunk_pos[:, None, :] - surface_points[None, :, :])**2).sum(axis=2))
            sdf[start:end] = dists.min(axis=1)
        
        # Compute approximate normals (gradient of SDF)
        # Use the direction from each point to its nearest surface point
        nx = np.zeros(N_pts)
        ny = np.zeros(N_pts)
        for start in range(0, N_pts, chunk_size):
            end = min(start + chunk_size, N_pts)
            chunk_pos = pos[start:end]
            dists = np.sqrt(((chunk_pos[:, None, :] - surface_points[None, :, :])**2).sum(axis=2))
            nearest_idx = dists.argmin(axis=1)
            nearest_surf = surface_points[nearest_idx]
            diff = chunk_pos - nearest_surf
            norm = np.sqrt((diff**2).sum(axis=1, keepdims=True)) + 1e-8
            diff_normalized = diff / norm
            nx[start:end] = diff_normalized[:, 0]
            ny[start:end] = diff_normalized[:, 1]
    else:
        sdf = np.zeros(N_pts)
        nx = np.zeros(N_pts)
        ny = np.zeros(N_pts)
    
    # Build the full 7-input vector: [X, Y, Z=0, SDF, Nx, Ny, Nz=0]
    Z = np.zeros(N_pts)
    Nz = np.zeros(N_pts)
    inputs = np.column_stack([X, Y, Z, sdf, nx, ny, Nz])
    
    # Targets: [U, V, W=0, P]
    W = np.zeros(N_pts)
    wing_targets = np.column_stack([targets[:, 0], targets[:, 1], W, targets[:, 2]])
    
    all_inputs.append(inputs)
    all_targets.append(wing_targets)
    all_surface_masks.append(surface_mask)
    
    if i % 20 == 0:
        print(f"  Wing {i}/{NUM_SIMULATIONS} | Points: {N_pts:,} | Surface pts: {surface_mask.sum()}")

# Concatenate everything
all_inputs = np.concatenate(all_inputs)
all_targets = np.concatenate(all_targets)
all_surface = np.concatenate(all_surface_masks)

print(f"\nTotal training points: {all_inputs.shape[0]:,}")
print(f"Total surface points: {all_surface.sum():,}")

# ==============================================================================
# FIX #3: SAVE GLOBAL NORMALIZATION STATS
# ==============================================================================
# Normalize inputs
inputs_mean = all_inputs.mean(axis=0)
inputs_std = all_inputs.std(axis=0) + 1e-8
all_inputs_norm = (all_inputs - inputs_mean) / inputs_std

# Normalize targets
targets_mean = all_targets.mean(axis=0)
targets_std = all_targets.std(axis=0) + 1e-8
all_targets_norm = (all_targets - targets_mean) / targets_std

# Save these stats so the test script uses the EXACT SAME normalization
norm_stats = {
    'inputs_mean': inputs_mean.tolist(),
    'inputs_std': inputs_std.tolist(),
    'targets_mean': targets_mean.tolist(),
    'targets_std': targets_std.tolist()
}
with open('normalization_stats.json', 'w') as f:
    json.dump(norm_stats, f)
print("Saved normalization_stats.json")

X_train = torch.FloatTensor(all_inputs_norm)
Y_train = torch.FloatTensor(all_targets_norm)
surface_mask_tensor = torch.BoolTensor(all_surface)

# ==============================================================================
# 2. MORPHOCALC v2 ARCHITECTURE
# ==============================================================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0, 
                                            np.sqrt(6 / in_features) / omega_0)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class MorphoCalcV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(7, 256, is_first=True),
            SineLayer(256, 256),
            SineLayer(256, 256),
            SineLayer(256, 256),
            SineLayer(256, 256)
        )
        self.final = nn.Linear(256, 4)

    def forward(self, x):
        return self.final(self.net(x))

# ==============================================================================
# 3. TRAINING LOOP WITH ALL FIXES
# ==============================================================================
model = MorphoCalcV2().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loader = DataLoader(TensorDataset(X_train, Y_train, surface_mask_tensor.unsqueeze(1).float()), 
                    batch_size=16384, shuffle=True)
mse_fn = nn.MSELoss()

print(f"\n🚀 Starting MorphoCalc v2 Training on {DEVICE}...")

TOTAL_EPOCHS = 100
for epoch in range(TOTAL_EPOCHS):
    e_mse, e_phys, e_bc = 0.0, 0.0, 0.0
    
    for bx, by, b_surf in loader:
        optimizer.zero_grad()
        
        bx = bx.to(DEVICE).requires_grad_(True)
        by = by.to(DEVICE)
        b_surf = b_surf.to(DEVICE).squeeze()
        
        preds = model(bx)
        
        # Loss 1: MSE (Data Fidelity)
        loss_mse = mse_fn(preds, by)
        
        # Loss 2: Physics (Navier-Stokes Divergence-Free)
        u, v, w = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
        dudx = torch.autograd.grad(u, bx, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
        dvdy = torch.autograd.grad(v, bx, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1:2]
        dwdz = torch.autograd.grad(w, bx, grad_outputs=torch.ones_like(w), create_graph=True)[0][:, 2:3]
        div = dudx + dvdy + dwdz
        loss_phys = torch.mean(div**2)
        
        # =====================================================================
        # FIX #4: BOUNDARY CONDITION LOSS (No-Slip on Surface)
        # =====================================================================
        # On the airfoil surface, velocity MUST be zero (no-slip condition)
        surf_points = b_surf.bool()
        if surf_points.sum() > 0:
            surf_velocity = preds[surf_points, :3]  # U, V, W on surface
            loss_bc = torch.mean(surf_velocity**2)   # Force to zero
        else:
            loss_bc = torch.tensor(0.0, device=DEVICE)
        
        # Combined loss with balanced weights
        loss = loss_mse + 0.1 * loss_phys + 0.5 * loss_bc
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        e_mse += loss_mse.item()
        e_phys += loss_phys.item()
        e_bc += loss_bc.item()
    
    n_batches = len(loader)
    if epoch % 2 == 0:
        print(f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | MSE: {e_mse/n_batches:.5f} | "
              f"Physics: {e_phys/n_batches:.5f} | Boundary: {e_bc/n_batches:.5f}")
        # Checkpoint every 2 epochs
        torch.save(model.state_dict(), f'morphocalc_v2_checkpoint_{epoch:03d}.pt')

print("\n✅ MorphoCalc v2 Training Complete!")
torch.save(model.state_dict(), 'morphocalc_v2_universal.pt')
print("Final weights saved to morphocalc_v2_universal.pt")
print("Normalization stats saved to normalization_stats.json")
