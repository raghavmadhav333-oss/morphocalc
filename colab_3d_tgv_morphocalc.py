"""
================================================================================
MorphoCalc 3D: Sparse Data Assimilation on 3D Taylor-Green Vortex
================================================================================
Author: Raghav Mukund
Project: MorphoCalc (Physics-Aware Representation Calculus)

What this does (honestly):
  - Uses the 3D Taylor-Green Vortex (TGV) at t=0 as the benchmark.
  - The TGV is the standard DNS benchmark for 3D turbulence transition (Re=1600).
  - We train on only 15% of 3D mesh points (sparse sampling).
  - We reconstruct the full 3D velocity field (u, v, w) at 100% of points.
  - We enforce 3D incompressibility: du/dx + dv/dy + dw/dz = 0.

Why 3D matters over 2D:
  - In 2D, vortex stretching (omega . grad u) is mathematically zero.
  - TRUE turbulence only exists in 3D. This is the physics boundary.
  - This is the step from proof-of-concept to genuine turbulence-relevant research.

Honest Limitations:
  - We use the analytical t=0 initial condition of the TGV.
  - This is NOT a turbulent simulation (turbulence develops at t > 0).
  - It IS the standard starting point for all DNS TGV benchmarks.
  - No temporal evolution. Steady-state spatial reconstruction only.
  - True turbulent DNS data requires JHTDB API (future work).

Instructions for Colab (NO pip install needed):
  1. Runtime -> Change runtime type -> T4 GPU
  2. Paste ENTIRE script into ONE cell and Run.
================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. GENERATE 3D TAYLOR-GREEN VORTEX DATA
#    Analytical initial condition (t=0).
#    This is the EXACT same initial condition used in all major DNS benchmarks
#    including the Johns Hopkins Turbulence Database Re=1600 dataset.
#    u =  sin(x) cos(y) cos(z)
#    v = -cos(x) sin(y) cos(z)
#    w =  0  (at t=0 by symmetry)
#    Domain: [0, 2pi]^3 periodic
# ==========================================
print("[*] Generating 3D Taylor-Green Vortex (t=0 analytical initial condition)...")
print("[*] Domain: [0, 2pi]^3 | This is the standard DNS turbulence benchmark.")

N_TOTAL = 80000   # Total 3D mesh points — manageable on T4 GPU

rng = np.random.default_rng(42)
# Random sparse sampling in the 3D periodic domain [0, 2pi]^3
px = rng.uniform(0, 2 * np.pi, N_TOTAL).astype(np.float32)
py = rng.uniform(0, 2 * np.pi, N_TOTAL).astype(np.float32)
pz = rng.uniform(0, 2 * np.pi, N_TOTAL).astype(np.float32)

# Exact analytical solution — this is ground truth, not approximation
vel_u = ( np.sin(px) * np.cos(py) * np.cos(pz)).astype(np.float32)
vel_v = (-np.cos(px) * np.sin(py) * np.cos(pz)).astype(np.float32)
vel_w = ( np.zeros_like(px)).astype(np.float32)   # Zero at t=0

# Verify incompressibility analytically:
# du/dx + dv/dy + dw/dz = cos(x)cos(y)cos(z) - cos(x)cos(y)cos(z) + 0 = 0 ✓
print(f"[*] Generated {N_TOTAL:,} 3D mesh points.")
print(f"[*] Velocity range: u=[{vel_u.min():.3f}, {vel_u.max():.3f}], "
      f"v=[{vel_v.min():.3f}, {vel_v.max():.3f}], w=[{vel_w.min():.3f}, {vel_w.max():.3f}]")
print(f"[*] Note: w=0 at t=0 by symmetry — this is physically correct.")

# ==========================================
# 2. SETUP DEVICE
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Training on: {device}")
if device.type == 'cpu':
    print("[!] WARNING: CPU mode. This will be slow (~30 min). Switch to T4 GPU in Runtime settings.")

# Stack into arrays
positions  = np.column_stack((px, py, pz))   # (N, 3)
velocities = np.column_stack((vel_u, vel_v, vel_w))  # (N, 3)

X_full = torch.tensor(positions,  dtype=torch.float32).to(device)
Y_full = torch.tensor(velocities, dtype=torch.float32).to(device)

# ==========================================
# 3. TRAIN/TEST SPLIT (15% SPARSE)
# ==========================================
TRAIN_FRACTION = 0.15
N = N_TOTAL
train_size = int(TRAIN_FRACTION * N)
indices    = torch.randperm(N, device=device)
train_idx  = indices[:train_size]
test_idx   = indices[train_size:]

X_train = X_full[train_idx]
Y_train = Y_full[train_idx]
X_test  = X_full[test_idx]
Y_test  = Y_full[test_idx]

print(f"[*] Train: {train_size:,} points ({TRAIN_FRACTION*100:.0f}%) | "
      f"Test: {N - train_size:,} points ({(1-TRAIN_FRACTION)*100:.0f}%)")

# ==========================================
# 4. NORMALIZATION
#    Coordinates: normalize to [-1, 1] for SIREN.
#    Velocities: global max-magnitude scaling to preserve 3D vector angles.
# ==========================================
x_min = X_full.min(dim=0)[0]
x_max = X_full.max(dim=0)[0]

def normalize_coords(X):
    return 2.0 * (X - x_min) / (x_max - x_min + 1e-8) - 1.0

X_train_norm = normalize_coords(X_train)
X_test_norm  = normalize_coords(X_test)
X_full_norm  = normalize_coords(X_full)

# Global max-magnitude scaling: preserves 3D vector angles
max_vel_mag   = torch.max(torch.norm(Y_full, dim=1)) + 1e-8
Y_train_norm  = Y_train / max_vel_mag
print(f"[*] Velocity normalization: global max-magnitude = {max_vel_mag.item():.4f} m/s")

# ==========================================
# 5. 3D SIREN ARCHITECTURE
#    Input: (x, y, z) — 3D coordinates
#    Output: (u, v, w) — 3D velocity vector
#    Larger hidden size (384) vs 2D (256) to handle extra dimension.
# ==========================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        self.linear   = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class MorphoCalcSIREN3D(nn.Module):
    """
    3D SIREN for sparse 3D flow field reconstruction.
    Input:  (x, y, z) normalized 3D coordinates
    Output: (u, v, w) velocity vector
    """
    def __init__(self, in_features=3, hidden_features=384, hidden_layers=6, out_features=3):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features))
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / 30
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        return self.net(coords), coords

print(f"[*] 3D SIREN: input=3, hidden=384x6, output=3 (u,v,w)")
total_params = sum(p.numel() for p in MorphoCalcSIREN3D().parameters())
print(f"[*] Total parameters: {total_params:,}")

# ==========================================
# 6. 3D LOSS FUNCTIONS
# ==========================================
def topological_alignment_loss_3d(pred_vel, true_vel):
    """
    3D Cosine similarity between predicted and true 3D velocity directions.
    Masked at stagnation points where the velocity magnitude is near-zero.
    """
    magnitude  = torch.norm(true_vel, dim=1, keepdim=True)
    mask       = (magnitude > 1e-4).float()
    v_hat      = true_vel / (magnitude + 1e-8)
    pred_mag   = torch.norm(pred_vel, dim=1, keepdim=True) + 1e-8
    pred_v_hat = pred_vel / pred_mag
    cosine_sim = torch.sum(v_hat * pred_v_hat, dim=1, keepdim=True)
    align_loss = (1.0 - cosine_sim) * mask
    return align_loss.sum() / (mask.sum() + 1e-8)


def physics_divergence_loss_3d(pred_vel, coords):
    """
    3D Incompressibility: du/dx + dv/dy + dw/dz = 0
    This is what makes 3D fundamentally different from 2D.
    In 3D, vortex stretching (omega . grad u) is non-zero — real turbulence physics.
    """
    U = pred_vel[:, 0:1]  # u
    V = pred_vel[:, 1:2]  # v
    W = pred_vel[:, 2:3]  # w

    dU = torch.autograd.grad(U, coords, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    dV = torch.autograd.grad(V, coords, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dW = torch.autograd.grad(W, coords, grad_outputs=torch.ones_like(W), create_graph=True)[0]

    # du/dx + dv/dy + dw/dz = 0
    divergence = dU[:, 0:1] + dV[:, 1:2] + dW[:, 2:3]
    return torch.mean(divergence ** 2)

# ==========================================
# 7. TRAINING
# ==========================================
NUM_ITERATIONS = 3000
BATCH_SIZE     = 2048   # Smaller than 2D (4096) due to 3D autograd memory cost
LEARNING_RATE  = 1e-4

model     = MorphoCalcSIREN3D(in_features=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS)

print(f"\n[*] Beginning {NUM_ITERATIONS}-Iteration 3D Training...")
print(f"[*] Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"[*] Note: 3D divergence loss computes 3 autograd passes — expect ~2x slower than 2D.")

start_time = time.time()

for step in range(NUM_ITERATIONS):
    actual_batch = min(BATCH_SIZE, X_train_norm.shape[0])
    idx     = torch.randperm(X_train_norm.shape[0], device=device)[:actual_batch]
    batch_X = X_train_norm[idx]
    batch_Y = Y_train_norm[idx]

    optimizer.zero_grad()
    pred_Y, coords_grad = model(batch_X)

    mse_loss  = nn.MSELoss()(pred_Y, batch_Y)
    topo_loss = topological_alignment_loss_3d(pred_Y, batch_Y)
    div_loss  = physics_divergence_loss_3d(pred_Y, coords_grad)
    total     = mse_loss + 0.5 * topo_loss + 0.1 * div_loss

    total.backward()
    optimizer.step()
    scheduler.step()

    if step % 250 == 0:
        print(f"Iter {step:04d} | MSE: {mse_loss.item():.6f} | "
              f"Topo: {topo_loss.item():.6f} | Div: {div_loss.item():.6f} | "
              f"Time: {time.time()-start_time:.1f}s")

print(f"[*] Training Complete! Total time: {time.time() - start_time:.1f}s")

# ==========================================
# 8. EVALUATION ON HELD-OUT 85% TEST SET
# ==========================================
print("\n[*] Evaluating on held-out 85% test set...")
model.eval()
with torch.no_grad():
    # Predict in chunks to avoid OOM on full test set
    chunk_size = 8192
    preds_list = []
    for i in range(0, X_test_norm.shape[0], chunk_size):
        chunk = X_test_norm[i:i+chunk_size]
        chunk_req = chunk.clone().detach().requires_grad_(False)
        pred_chunk, _ = model(chunk_req)
        preds_list.append(pred_chunk)
    Y_pred_norm = torch.cat(preds_list, dim=0)

# Denormalize
Y_pred = Y_pred_norm * max_vel_mag

# Relative L2 Error on full 3D velocity magnitude
true_mag = torch.norm(Y_test, dim=1)
pred_mag_val = torch.norm(Y_pred, dim=1)
rel_l2 = torch.norm(pred_mag_val - true_mag) / (torch.norm(true_mag) + 1e-8)

# 3D Angular Error (degrees)
true_hat = Y_test / (true_mag.unsqueeze(1) + 1e-8)
pred_hat = Y_pred / (pred_mag_val.unsqueeze(1) + 1e-8)
stag_mask = true_mag > 1e-4
cosine_vals = torch.sum(true_hat[stag_mask] * pred_hat[stag_mask], dim=1).clamp(-1.0, 1.0)
angular_err_deg = torch.mean(torch.acos(cosine_vals)) * (180.0 / np.pi)

print(f"\n{'='*60}")
print(f"  3D MorphoCalc Results (85% unseen test points)")
print(f"{'='*60}")
print(f"  Relative L2 Error (velocity magnitude): {rel_l2.item():.4f} ({rel_l2.item()*100:.2f}%)")
print(f"  Mean Angular Error (velocity direction): {angular_err_deg.item():.4f}°")
print(f"{'='*60}")
print(f"  2D Baseline (Joukowski): L2=0.0062, Angular=0.05°")
print(f"  3D Extension (TGV t=0): L2={rel_l2.item():.4f}, Angular={angular_err_deg.item():.4f}°")
print(f"{'='*60}")

# ==========================================
# 9. VISUALIZATION (2D SLICES THROUGH 3D FIELD)
#    We show the z=pi plane (mid-domain cross-section).
#    This is the standard way to visualize 3D flow fields.
# ==========================================
print("\n[*] Generating visualization (z=pi cross-section slice)...")

# Create a 2D grid at z=pi for visualization
N_VIZ = 64
x_viz = np.linspace(0, 2 * np.pi, N_VIZ, dtype=np.float32)
y_viz = np.linspace(0, 2 * np.pi, N_VIZ, dtype=np.float32)
XX, YY = np.meshgrid(x_viz, y_viz)
ZZ = np.ones_like(XX) * np.pi  # z = pi (center of domain)

# Ground truth on this slice
GT_u = np.sin(XX) * np.cos(YY) * np.cos(ZZ)
GT_v = -np.cos(XX) * np.sin(YY) * np.cos(ZZ)
GT_mag = np.sqrt(GT_u**2 + GT_v**2)

# Model prediction on this slice
viz_pts = np.column_stack((XX.ravel(), YY.ravel(), ZZ.ravel()))
viz_tensor = torch.tensor(viz_pts, dtype=torch.float32).to(device)
viz_norm = normalize_coords(viz_tensor)
with torch.no_grad():
    viz_pred, _ = model(viz_norm)
viz_pred = (viz_pred * max_vel_mag).cpu().numpy()
PRED_u = viz_pred[:, 0].reshape(N_VIZ, N_VIZ)
PRED_v = viz_pred[:, 1].reshape(N_VIZ, N_VIZ)
PRED_mag = np.sqrt(PRED_u**2 + PRED_v**2)

# Absolute error
ERR_mag = np.abs(PRED_mag - GT_mag)
slice_l2 = np.linalg.norm(PRED_mag - GT_mag) / (np.linalg.norm(GT_mag) + 1e-8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"MorphoCalc 3D: Taylor-Green Vortex | z=π Cross-Section\n"
             f"15% Sparse → Full 3D Reconstruction | "
             f"Rel. L2={rel_l2.item():.4f} | Angular={angular_err_deg.item():.2f}°",
             fontsize=12, fontweight='bold')

vmin, vmax = GT_mag.min(), GT_mag.max()
cmap = 'RdYlBu_r'

im0 = axes[0].contourf(XX, YY, GT_mag, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
axes[0].set_title('Ground Truth\n(Analytical TGV, z=π slice)', fontsize=10)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0], label='Velocity Magnitude (m/s)')

im1 = axes[1].contourf(XX, YY, PRED_mag, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
axes[1].set_title('MorphoCalc Reconstruction\n(15% Sparse → Full 3D Field)', fontsize=10)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
plt.colorbar(im1, ax=axes[1], label='Velocity Magnitude (m/s)')

im2 = axes[2].contourf(XX, YY, ERR_mag, levels=50, cmap='hot_r')
axes[2].set_title(f'Absolute Error\n(Slice L2={slice_l2:.4f})', fontsize=10)
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
plt.colorbar(im2, ax=axes[2], label='Error (m/s)')

plt.tight_layout()
plt.savefig('morphocalc_3d_tgv_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("[*] Plot saved: morphocalc_3d_tgv_results.png")
print("[*] Download it from the Colab Files panel (left sidebar).")

# ==========================================
# 10. SUMMARY for Publication POSTER
# ==========================================
print(f"""
{'='*60}
Research RESULT SUMMARY
{'='*60}
Experiment: 3D Taylor-Green Vortex Sparse Reconstruction
Data:       Analytical TGV at t=0 (standard DNS initial condition)
Domain:     [0, 2pi]^3, {N_TOTAL:,} random 3D mesh points
Training:   {train_size:,} points (15% sparse)
Test:       {N - train_size:,} unseen points (85%)

Network:    3D SIREN (in=3, hidden=384x6, out=3)
Parameters: {total_params:,}
Iterations: {NUM_ITERATIONS}

METRIC              VALUE
---------           -----
Rel. L2 Error:      {rel_l2.item():.4f} ({rel_l2.item()*100:.2f}%)
Angular Error:      {angular_err_deg.item():.4f} degrees

Physics constraint: 3D divergence du/dx + dv/dy + dw/dz = 0 (enforced via autograd)
Key difference from 2D: 3D vortex stretching (omega . grad u) is non-zero here.
{'='*60}
""")
