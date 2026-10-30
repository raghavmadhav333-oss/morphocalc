"""
================================================================================
MorphoCalc: Sparse Data Assimilation (Control Run - NO Topo Loss)
================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# FIXED SEED FOR FAIR COMPARISON
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. LOAD DATA
# ==========================================
pos_x = pos_y = vel_u = vel_v = sdf = name = None

print("[*] No uploaded file found. Trying wget for Real CFD Navier-Stokes data...")
import urllib.request
import scipy.io as io

mat_url = "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/cylinder_nektar_wake.mat"
outfile = "/content/cylinder.mat"
try:
    urllib.request.urlretrieve(mat_url, outfile)
    data = io.loadmat(outfile)
    
    pos_x = data['X_star'][:, 0].astype(np.float32)
    pos_y = data['X_star'][:, 1].astype(np.float32)
    
    vel_u = data['U_star'][:, 0, 100].astype(np.float32)
    vel_v = data['U_star'][:, 1, 100].astype(np.float32)
    
    name  = "Navier_Stokes_Cylinder_Wake"
    print(f"[*] wget success: {name} | N={len(pos_x):,}")
except Exception as e:
    print(f"[!] wget failed ({e}).")

# ==========================================
# 2. SETUP DEVICE & DATA
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Training on device: {device}")

positions  = np.column_stack((pos_x, pos_y))    # (N, 2)
velocities = np.column_stack((vel_u, vel_v))    # (N, 2)
N = len(pos_x)

X_full = torch.tensor(positions,  dtype=torch.float32).to(device)
Y_full = torch.tensor(velocities, dtype=torch.float32).to(device)

# ==========================================
# 3. TRAIN/TEST SPLIT (5% SPARSE)
# ==========================================
TRAIN_FRACTION = 0.05
train_size = int(TRAIN_FRACTION * N)
indices   = torch.randperm(N, device=device)
train_idx = indices[:train_size]

X_train = X_full[train_idx]
Y_train = Y_full[train_idx]

print(f"[*] Train/Test Split: {train_size:,} training points ({TRAIN_FRACTION*100:.0f}%) "
      f"| {N - train_size:,} test points ({(1-TRAIN_FRACTION)*100:.0f}%)")

# ==========================================
# 4. NORMALIZATION
# ==========================================
x_min = X_full.min(dim=0)[0]
x_max = X_full.max(dim=0)[0]

def normalize_coords(X):
    return 2.0 * (X - x_min) / (x_max - x_min + 1e-8) - 1.0

X_train_norm = normalize_coords(X_train)
X_full_norm  = normalize_coords(X_full)

max_vel_magnitude = torch.max(torch.norm(Y_full, dim=1)) + 1e-8
Y_train_norm = Y_train / max_vel_magnitude
print(f"[*] Velocity Normalization: Global Max-Magnitude = {max_vel_magnitude.item():.4f} m/s")

# ==========================================
# 5. SIREN ARCHITECTURE
# ==========================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=3):
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


class MorphoCalcSIREN(nn.Module):
    def __init__(self, in_features=2, hidden_features=1024, hidden_layers=8, out_features=2):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features))
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / 3
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        return self.net(coords), coords

# ==========================================
# 6. LOSS FUNCTIONS
# ==========================================
def topological_alignment_loss(pred_vel, true_vel):
    magnitude  = torch.norm(true_vel, dim=1, keepdim=True)
    mask       = (magnitude > 1e-4).float()
    v_hat      = true_vel / (magnitude + 1e-8)
    pred_mag   = torch.norm(pred_vel, dim=1, keepdim=True) + 1e-8
    pred_v_hat = pred_vel / pred_mag
    cosine_sim = torch.sum(v_hat * pred_v_hat, dim=1, keepdim=True)
    align_loss = (1.0 - cosine_sim) * mask
    return align_loss.sum() / (mask.sum() + 1e-8)

def physics_divergence_loss(pred_vel, coords):
    U  = pred_vel[:, 0:1]
    V  = pred_vel[:, 1:2]
    dU = torch.autograd.grad(U, coords, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    dV = torch.autograd.grad(V, coords, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    return torch.mean((dU[:, 0:1] + dV[:, 1:2]) ** 2)

# ==========================================
# 7. TRAINING
# ==========================================
NUM_ITERATIONS = 20000
BATCH_SIZE     = 16384
LEARNING_RATE  = 1e-4

model     = MorphoCalcSIREN(in_features=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS)

print(f"[*] Beginning {NUM_ITERATIONS}-Iteration Training (CONTROL - TOPO LOSS OFF)...")
start_time = time.time()

for step in range(NUM_ITERATIONS):
    actual_batch = min(BATCH_SIZE, X_train_norm.shape[0])
    idx     = torch.randperm(X_train_norm.shape[0], device=device)[:actual_batch]
    batch_X = X_train_norm[idx]
    batch_Y = Y_train_norm[idx]

    optimizer.zero_grad()
    pred_Y, coords_grad = model(batch_X)

    mse_loss  = nn.MSELoss()(pred_Y, batch_Y)
    topo_loss = topological_alignment_loss(pred_Y, batch_Y)
    div_loss  = physics_divergence_loss(pred_Y, coords_grad)
    
    # CONTROL RUN: TOPO LOSS IS MULTIPLIED BY 0.0
    total     = mse_loss + 0.0 * topo_loss + 0.1 * div_loss

    total.backward()
    optimizer.step()
    scheduler.step()

    if step % 250 == 0:
        print(f"Iter {step:04d} | MSE: {mse_loss.item():.6f} | "
              f"Topo (Ignored): {topo_loss.item():.6f} | Div: {div_loss.item():.6f} | "
              f"Time: {time.time()-start_time:.1f}s")

print(f"[*] Training Complete! Total time: {time.time() - start_time:.1f}s")

# ==========================================
# 8. INFERENCE & VISUALIZATION
# ==========================================
print("[*] Reconstructing full velocity field...")
with torch.no_grad():
    pred_norm, _ = model(X_full_norm)
    pred_vel_full = pred_norm * max_vel_magnitude

pred_u = pred_vel_full[:, 0].cpu().numpy()
pred_v = pred_vel_full[:, 1].cpu().numpy()
true_u = vel_u
true_v = vel_v

pred_mag_np = np.sqrt(pred_u**2 + pred_v**2)
true_mag_np = np.sqrt(true_u**2 + true_v**2)

l2_error = np.linalg.norm(pred_mag_np - true_mag_np) / (np.linalg.norm(true_mag_np) + 1e-8)

mag_mask    = true_mag_np > 1e-4
true_u_hat  = true_u[mag_mask] / (true_mag_np[mag_mask] + 1e-8)
true_v_hat  = true_v[mag_mask] / (true_mag_np[mag_mask] + 1e-8)
pred_u_hat  = pred_u[mag_mask] / (pred_mag_np[mag_mask] + 1e-8)
pred_v_hat  = pred_v[mag_mask] / (pred_mag_np[mag_mask] + 1e-8)
cosine_vals = np.clip(true_u_hat*pred_u_hat + true_v_hat*pred_v_hat, -1, 1)
angular_err = np.mean(np.degrees(np.arccos(cosine_vals)))

print(f"\n{'='*60}")
print(f"  CONTROL RUN (Topo OFF) Results")
print(f"{'='*60}")
print(f"  Relative L2 Error:   {l2_error:.6f}  ({l2_error*100:.2f}%)")
print(f"  Mean Angular Error:  {angular_err:.4f} degrees")
print(f"{'='*60}\n")
