# ==============================================================================
# MORPHOCALC v2 - UNSEEN WING TESTING SCRIPT (MATCHES v2 TRAINING)
# ==============================================================================

!pip install torch_geometric

import torch
import torch.nn as nn
from torch_geometric.datasets import AirfRANS
import numpy as np
import matplotlib.pyplot as plt
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. ARCHITECTURE (Must match training exactly)
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
# 2. LOAD TRAINED BRAIN + NORMALIZATION STATS
# ==============================================================================
# 🛑 UPDATE THESE PATHS to wherever your files are saved in Kaggle!
WEIGHTS_PATH = '/kaggle/input/YOUR_DATASET/morphocalc_v2_universal.pt'
STATS_PATH = '/kaggle/input/YOUR_DATASET/normalization_stats.json'

model = MorphoCalcV2().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print("✅ MorphoCalc v2 brain loaded!")

# FIX #3: Load the EXACT normalization stats from training
with open(STATS_PATH, 'r') as f:
    stats = json.load(f)
inputs_mean = np.array(stats['inputs_mean'])
inputs_std = np.array(stats['inputs_std'])
targets_mean = np.array(stats['targets_mean'])
targets_std = np.array(stats['targets_std'])
print("✅ Normalization stats loaded!")

# ==============================================================================
# 3. LOAD UNSEEN WING & COMPUTE ITS SDF/NORMALS
# ==============================================================================
print("Loading unseen wing...")
dataset = AirfRANS(root='./AirfRANS_data', task='full', train=True)
unseen = dataset[101]  # NEVER SEEN DURING TRAINING

pos = unseen.pos.numpy()
targets_real = unseen.y.numpy()

X = pos[:, 0]
Y = pos[:, 1]
U_real = targets_real[:, 0]
V_real = targets_real[:, 1]
N_pts = X.shape[0]

# Compute SDF and Normals for the unseen wing (same method as training)
U_mag = np.sqrt(U_real**2 + V_real**2)
surface_threshold = 0.01 * np.max(U_mag)
surface_mask = U_mag < surface_threshold
surface_points = pos[surface_mask]

sdf = np.zeros(N_pts)
nx = np.zeros(N_pts)
ny = np.zeros(N_pts)

if len(surface_points) > 10:
    chunk_size = 5000
    for start in range(0, N_pts, chunk_size):
        end = min(start + chunk_size, N_pts)
        chunk_pos = pos[start:end]
        dists = np.sqrt(((chunk_pos[:, None, :] - surface_points[None, :, :])**2).sum(axis=2))
        sdf[start:end] = dists.min(axis=1)
        nearest_idx = dists.argmin(axis=1)
        nearest_surf = surface_points[nearest_idx]
        diff = chunk_pos - nearest_surf
        norm = np.sqrt((diff**2).sum(axis=1, keepdims=True)) + 1e-8
        diff_normalized = diff / norm
        nx[start:end] = diff_normalized[:, 0]
        ny[start:end] = diff_normalized[:, 1]

# Build input vector with REAL geometry data
inputs = np.column_stack([X, Y, np.zeros(N_pts), sdf, nx, ny, np.zeros(N_pts)])

# Normalize using the SAME stats from training
inputs_norm = (inputs - inputs_mean) / inputs_std
X_test = torch.FloatTensor(inputs_norm).to(DEVICE)

# ==============================================================================
# 4. PREDICT & VISUALIZE (Research-READY)
# ==============================================================================
print("Predicting aerodynamics for Unseen Wing #101...")
with torch.no_grad():
    preds_norm = model(X_test).cpu().numpy()

# Un-normalize using the EXACT training stats
preds = preds_norm * targets_std + targets_mean
U_pred = preds[:, 0]

# Downsample for fast plotting
skip = 5
X_fast, Y_fast = X[::skip], Y[::skip]
U_pred_fast = U_pred[::skip]
U_real_fast = U_real[::skip]

# Error metrics
error = np.abs(U_real_fast - U_pred_fast)
mae = np.mean(error)
rmse = np.sqrt(np.mean(error**2))
rel_error = mae / (np.abs(U_real_fast).mean() + 1e-8) * 100

# Lock color scales
vmin = min(U_real_fast.min(), U_pred_fast.min())
vmax = max(U_real_fast.max(), U_pred_fast.max())

plt.figure(figsize=(22, 7))
plt.suptitle(f'MorphoCalc v2 — Generalization on UNSEEN Wing #101\n'
             f'MAE: {mae:.2f}  |  RMSE: {rmse:.2f}  |  Relative Error: {rel_error:.1f}%', 
             fontsize=16, fontweight='bold')

plt.subplot(1, 3, 1)
plt.scatter(X_fast, Y_fast, c=U_real_fast, cmap='jet', s=2, vmin=vmin, vmax=vmax)
plt.colorbar(label='Velocity U (m/s)')
plt.title('Supercomputer Ground Truth', fontsize=13)
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.scatter(X_fast, Y_fast, c=U_pred_fast, cmap='jet', s=2, vmin=vmin, vmax=vmax)
plt.colorbar(label='Velocity U (m/s)')
plt.title('MorphoCalc v2 AI Prediction', fontsize=13)
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.scatter(X_fast, Y_fast, c=error, cmap='Reds', s=2, vmin=0, vmax=vmax*0.15)
plt.colorbar(label='Absolute Error')
plt.title('Error Differential', fontsize=13)
plt.axis('equal')

plt.tight_layout()
plt.savefig('morphocalc_v2_unseen_wing_101.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved to morphocalc_v2_unseen_wing_101.png")
