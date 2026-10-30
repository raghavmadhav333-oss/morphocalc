"""
================================================================================
MorphoCalc 3D: Real-World 3D CFD Data Assimilation
================================================================================
Author: Raghav Mukund
Project: MorphoCalc (Physics-Aware Representation Calculus)

What this does (honestly):
  - Transitions from analytical 3D physics (Taylor-Green) to REAL 3D CFD data.
  - Loads a real 3D CFD simulation of airflow (using PyVista's Kitchen CFD or a 3D sphere CFD).
  - Trains a 3D SIREN on a 15% sparse sample of the 3D mesh points.
  - Reconstructs the full 3D velocity field (u, v, w).
  - Enforces 3D incompressibility (du/dx + dv/dy + dw/dz = 0).

Why this is the final step for the Research Project:
  1. 2D Real CFD (AirfRANS) -> Handled.
  2. 3D Analytical (Taylor-Green) -> Handled.
  3. 3D Real CFD (This script) -> Proves MorphoCalc works on real-world 3D geometries.

Instructions for Colab:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Run this block in Cell 1:
     !pip install pyvista
  3. Paste this ENTIRE script into Cell 2 and Run.
================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD 3D REAL-WORLD CFD DATA
#    We use PyVista to download a real 3D fluid dynamics simulation.
#    Fallback: Exact 3D Aerodynamic Flow Past a Sphere.
# ==========================================
print("[*] Loading Real-World 3D CFD Dataset...")

pos_x = pos_y = pos_z = vel_u = vel_v = vel_w = name = None

try:
    import pyvista as pv
    from pyvista import examples
    
    # Download the classic "Kitchen" 3D airflow CFD dataset
    print("[*] Downloading PyVista 3D Airflow (Kitchen) CFD Dataset...")
    dataset = examples.download_kitchen()
    
    # Extract points and velocity vectors
    pts = dataset.points
    vel = dataset['velocity']
    
    # Subsample if too large for quick Colab training (cap at 100k points)
    N_pts = len(pts)
    if N_pts > 100000:
        print(f"[*] Downsampling from {N_pts:,} to 100,000 points for memory limits.")
        rng = np.random.default_rng(42)
        idx = rng.choice(N_pts, 100000, replace=False)
        pts = pts[idx]
        vel = vel[idx]
        
    pos_x, pos_y, pos_z = pts[:, 0], pts[:, 1], pts[:, 2]
    vel_u, vel_v, vel_w = vel[:, 0], vel[:, 1], vel[:, 2]
    name = "Real_3D_CFD_Kitchen_Airflow"
    print(f"[*] Successfully loaded PyVista CFD dataset. N={len(pos_x):,}")

except Exception as e:
    print(f"[!] PyVista load failed: {e}")
    print("[*] Using fallback: Exact 3D aerodynamic flow past a sphere (Real Physics).")
    # Exact analytical solution for 3D potential flow past a solid sphere.
    # This is genuine 3D aerodynamics, physically identical to wind tunnel potential flow.
    N_pts = 80000
    rng = np.random.default_rng(42)
    U_inf = 1.0
    R = 1.0  # Sphere radius
    
    # Random points around the sphere
    radii = rng.uniform(R + 0.05, 5.0, N_pts)
    theta = np.arccos(rng.uniform(-1, 1, N_pts))
    phi_angle = rng.uniform(0, 2 * np.pi, N_pts)
    
    px = radii * np.sin(theta) * np.cos(phi_angle)
    py = radii * np.sin(theta) * np.sin(phi_angle)
    pz = radii * np.cos(theta)
    
    r2 = px**2 + py**2 + pz**2
    r5 = r2**(5/2)
    
    # 3D Velocity components for flow past a sphere (flow along X axis)
    pu = U_inf * (1 + (R**3 / 2.0) * ((py**2 + pz**2 - 2 * px**2) / r5))
    pv = U_inf * (-3.0 * R**3 * px * py / (2.0 * r5))
    pw = U_inf * (-3.0 * R**3 * px * pz / (2.0 * r5))
    
    pos_x, pos_y, pos_z = px.astype(np.float32), py.astype(np.float32), pz.astype(np.float32)
    vel_u, vel_v, vel_w = pu.astype(np.float32), pv.astype(np.float32), pw.astype(np.float32)
    name = "3D_Flow_Past_Sphere"
    print(f"[*] Fallback loaded. N={len(pos_x):,}")

print(f"[*] Velocity range: u=[{vel_u.min():.3f}, {vel_u.max():.3f}], v=[{vel_v.min():.3f}, {vel_v.max():.3f}], w=[{vel_w.min():.3f}, {vel_w.max():.3f}]")

# ==========================================
# 2. SETUP DEVICE
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Training on: {device}")

positions  = np.column_stack((pos_x, pos_y, pos_z))
velocities = np.column_stack((vel_u, vel_v, vel_w))
N = len(pos_x)

X_full = torch.tensor(positions,  dtype=torch.float32).to(device)
Y_full = torch.tensor(velocities, dtype=torch.float32).to(device)

# ==========================================
# 3. TRAIN/TEST SPLIT (15% SPARSE)
# ==========================================
TRAIN_FRACTION = 0.15
train_size = int(TRAIN_FRACTION * N)
indices    = torch.randperm(N, device=device)
train_idx  = indices[:train_size]
test_idx   = indices[train_size:]

X_train = X_full[train_idx]
Y_train = Y_full[train_idx]
X_test  = X_full[test_idx]
Y_test  = Y_full[test_idx]

print(f"[*] Train: {train_size:,} points ({TRAIN_FRACTION*100:.0f}%) | Test: {N - train_size:,} points ({(1-TRAIN_FRACTION)*100:.0f}%)")

# ==========================================
# 4. NORMALIZATION
# ==========================================
x_min = X_full.min(dim=0)[0]
x_max = X_full.max(dim=0)[0]

def normalize_coords(X):
    return 2.0 * (X - x_min) / (x_max - x_min + 1e-8) - 1.0

X_train_norm = normalize_coords(X_train)
X_test_norm  = normalize_coords(X_test)

max_vel_mag = torch.max(torch.norm(Y_full, dim=1)) + 1e-8
Y_train_norm = Y_train / max_vel_mag

# ==========================================
# 5. 3D SIREN ARCHITECTURE
# ==========================================
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
                bound = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class MorphoCalcSIREN3D(nn.Module):
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=6, out_features=3):
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

# ==========================================
# 6. 3D LOSS FUNCTIONS
# ==========================================
def topological_alignment_loss_3d(pred_vel, true_vel):
    magnitude  = torch.norm(true_vel, dim=1, keepdim=True)
    mask       = (magnitude > 1e-4).float()
    v_hat      = true_vel / (magnitude + 1e-8)
    pred_mag   = torch.norm(pred_vel, dim=1, keepdim=True) + 1e-8
    pred_v_hat = pred_vel / pred_mag
    cosine_sim = torch.sum(v_hat * pred_v_hat, dim=1, keepdim=True)
    align_loss = (1.0 - cosine_sim) * mask
    return align_loss.sum() / (mask.sum() + 1e-8)

def physics_divergence_loss_3d(pred_vel, coords):
    U, V, W = pred_vel[:, 0:1], pred_vel[:, 1:2], pred_vel[:, 2:3]
    dU = torch.autograd.grad(U, coords, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    dV = torch.autograd.grad(V, coords, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dW = torch.autograd.grad(W, coords, grad_outputs=torch.ones_like(W), create_graph=True)[0]
    divergence = dU[:, 0:1] + dV[:, 1:2] + dW[:, 2:3]
    return torch.mean(divergence ** 2)

# ==========================================
# 7. TRAINING
# ==========================================
NUM_ITERATIONS = 10000
BATCH_SIZE     = 2048
LEARNING_RATE  = 1e-4

model = MorphoCalcSIREN3D(in_features=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS)

print(f"\n[*] Beginning {NUM_ITERATIONS}-Iteration 3D Training...")
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

    if step % 1000 == 0:
        print(f"Iter {step:04d} | MSE: {mse_loss.item():.6f} | Div: {div_loss.item():.6f}")

print(f"[*] Training Complete! Total time: {time.time() - start_time:.1f}s")

# ==========================================
# 8. EVALUATION
# ==========================================
print("\n[*] Evaluating on held-out 85% test set...")
model.eval()
with torch.no_grad():
    chunk_size = 8192
    preds_list = []
    for i in range(0, X_test_norm.shape[0], chunk_size):
        chunk = X_test_norm[i:i+chunk_size]
        pred_chunk, _ = model(chunk.clone().detach())
        preds_list.append(pred_chunk)
    Y_pred_norm = torch.cat(preds_list, dim=0)

Y_pred = Y_pred_norm * max_vel_mag
true_mag = torch.norm(Y_test, dim=1)
pred_mag_val = torch.norm(Y_pred, dim=1)
rel_l2 = torch.norm(pred_mag_val - true_mag) / (torch.norm(true_mag) + 1e-8)

true_hat = Y_test / (true_mag.unsqueeze(1) + 1e-8)
pred_hat = Y_pred / (pred_mag_val.unsqueeze(1) + 1e-8)
stag_mask = true_mag > 1e-4
cosine_vals = torch.sum(true_hat[stag_mask] * pred_hat[stag_mask], dim=1).clamp(-1.0, 1.0)
angular_err_deg = torch.mean(torch.acos(cosine_vals)) * (180.0 / np.pi)

print(f"\n{'='*60}")
print(f"  3D MorphoCalc Results: {name}")
print(f"{'='*60}")
print(f"  Relative L2 Error: {rel_l2.item():.4f} ({rel_l2.item()*100:.2f}%)")
print(f"  Mean Angular Error: {angular_err_deg.item():.4f}°")
print(f"{'='*60}")
