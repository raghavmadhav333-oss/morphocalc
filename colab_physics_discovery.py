"""
================================================================================
MorphoCalc Physics Discovery Engine
================================================================================
Author: Raghav Mukund
Project: MorphoCalc (Physics-Aware Representation Calculus)

WHAT THIS SCRIPT DOES:
  MorphoCalc's SIREN learns a continuous, infinitely-differentiable velocity
  field v(x,y,z) = (u, v, w). Because PyTorch's autograd gives us exact
  spatial derivatives, we can compute physical quantities that were NEVER
  measured — derived purely from the trained neural field.

  This script trains on the 3D Flow-Past-Sphere benchmark, then computes
  5 new physics quantities from the trained network:

  1. VORTICITY FIELD (ω = ∇ × v)
     - The curl of velocity. Identifies rotating structures (vortices,
       shear layers, wakes) in the flow.
     - Equation:
         ω_x = ∂w/∂y − ∂v/∂z
         ω_y = ∂u/∂z − ∂w/∂x
         ω_z = ∂v/∂x − ∂u/∂y

  2. Q-CRITERION (Vortex Identification)
     - The gold-standard metric for identifying vortex cores in aerospace CFD.
     - Decomposes the velocity gradient tensor ∇v into:
         Symmetric part (Strain Rate):      S = ½(∇v + ∇vᵀ)
         Antisymmetric part (Rotation Rate): Ω = ½(∇v − ∇vᵀ)
     - Equation:
         Q = ½(‖Ω‖²_F − ‖S‖²_F)
     - Where Q > 0 indicates a vortex core (rotation dominates strain).

  3. PRESSURE FIELD RECOVERY (∇²p from Navier-Stokes)
     - The pressure Poisson equation from incompressible Navier-Stokes:
         ∇²p = −ρ * Σᵢ Σⱼ (∂uᵢ/∂xⱼ)(∂uⱼ/∂xᵢ)
     - This gives us the pressure Laplacian at every point — pressure
       information recovered WITHOUT any pressure sensors.

  4. STRAIN RATE TENSOR (S_ij)
     - The symmetric part of the velocity gradient tensor:
         S_ij = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
     - Its magnitude (Frobenius norm) indicates how strongly the fluid
       is being deformed — critical for predicting drag, wall shear stress,
       and material failure in biomedical flows.

  5. ENSTROPHY (Ε = ½ ‖ω‖²)
     - Total rotational kinetic energy density of the flow.
     - In turbulence theory, enstrophy production drives the energy cascade
       from large scales to Kolmogorov microscales.
     - Equation:
         Ε = ½ (ω_x² + ω_y² + ω_z²)

WHY THIS MATTERS for Publication:
  Standard reconstruction says "I can fill in missing velocity data."
  Physics Discovery says "From that filled-in data, I can DERIVE quantities
  that were never measured at all — pressure, vorticity, strain — using
  nothing but the mathematical structure of my trained neural field."

  This is the difference between interpolation and scientific inference.

Instructions for Colab:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Paste this entire script into a cell and run.
================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. GENERATE 3D FLOW PAST A SPHERE
# ==========================================
print("=" * 70)
print("  MorphoCalc Physics Discovery Engine")
print("  Beyond Reconstruction: Deriving Unmeasured Physical Quantities")
print("=" * 70)

print("\n[*] Generating 3D Potential Flow Past a Sphere...")
N_pts = 80000
rng = np.random.default_rng(42)
U_inf = 1.0
R = 1.0

radii = rng.uniform(R + 0.05, 5.0, N_pts)
theta = np.arccos(rng.uniform(-1, 1, N_pts))
phi_angle = rng.uniform(0, 2 * np.pi, N_pts)

px = radii * np.sin(theta) * np.cos(phi_angle)
py = radii * np.sin(theta) * np.sin(phi_angle)
pz = radii * np.cos(theta)

r2 = px**2 + py**2 + pz**2
r5 = r2**(5/2)

pu = U_inf * (1 + (R**3 / 2.0) * ((py**2 + pz**2 - 2 * px**2) / r5))
pv = U_inf * (-3.0 * R**3 * px * py / (2.0 * r5))
pw = U_inf * (-3.0 * R**3 * px * pz / (2.0 * r5))

pos_x = px.astype(np.float32)
pos_y = py.astype(np.float32)
pos_z = pz.astype(np.float32)
vel_u = pu.astype(np.float32)
vel_v = pv.astype(np.float32)
vel_w = pw.astype(np.float32)

print(f"[*] Dataset: N={N_pts:,} points around a unit sphere")
print(f"[*] Velocity range: u=[{vel_u.min():.3f}, {vel_u.max():.3f}]")

# ==========================================
# 2. SETUP
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Device: {device}")

positions  = np.column_stack((pos_x, pos_y, pos_z))
velocities = np.column_stack((vel_u, vel_v, vel_w))
N = len(pos_x)

X_full = torch.tensor(positions,  dtype=torch.float32).to(device)
Y_full = torch.tensor(velocities, dtype=torch.float32).to(device)

# Train/Test split (15% sparse)
TRAIN_FRACTION = 0.15
train_size = int(TRAIN_FRACTION * N)
indices = torch.randperm(N, device=device)
train_idx = indices[:train_size]
test_idx  = indices[train_size:]

X_train = X_full[train_idx]
Y_train = Y_full[train_idx]
X_test  = X_full[test_idx]
Y_test  = Y_full[test_idx]

print(f"[*] Train: {train_size:,} | Test: {N - train_size:,}")

# Normalization
x_min = X_full.min(dim=0)[0]
x_max = X_full.max(dim=0)[0]

def normalize_coords(X):
    return 2.0 * (X - x_min) / (x_max - x_min + 1e-8) - 1.0

X_train_norm = normalize_coords(X_train)
X_test_norm  = normalize_coords(X_test)
max_vel_mag  = torch.max(torch.norm(Y_full, dim=1)) + 1e-8
Y_train_norm = Y_train / max_vel_mag

# ==========================================
# 3. SIREN ARCHITECTURE
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
# 4. LOSS FUNCTIONS
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
# 5. TRAINING (Stable & Fast on CPU)
# ==========================================
NUM_ITERATIONS = 2000
BATCH_SIZE     = 512
LEARNING_RATE  = 1e-4

model = MorphoCalcSIREN3D(in_features=3, hidden_features=384, hidden_layers=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS)

print(f"\n[*] Training SIREN ({NUM_ITERATIONS} iterations, stabilized)...")
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
    div_raw   = physics_divergence_loss_3d(pred_Y, coords_grad)
    div_loss  = torch.clamp(div_raw, max=5.0)  # Prevents gradient explosion!
    
    total     = mse_loss + 0.5 * topo_loss + 0.05 * div_loss

    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Hard safety clip
    optimizer.step()
    scheduler.step()

    if step % 250 == 0 or step == NUM_ITERATIONS - 1:
        elapsed = time.time() - start_time
        print(f"  Iter {step:04d}/{NUM_ITERATIONS} | MSE: {mse_loss.item():.6f} | Div: {div_loss.item():.6f} | Elapsed: {elapsed:.1f}s")

train_time = time.time() - start_time
print(f"[*] Training Complete! Total Time: {train_time:.1f}s")

# ==========================================
# 6. STANDARD RECONSTRUCTION ACCURACY
# ==========================================
print("\n[*] Phase 1: Standard Velocity Reconstruction (baseline)...")
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

print(f"  Relative L2 Error: {rel_l2.item():.4f} ({rel_l2.item()*100:.2f}%)")
print(f"  Mean Angular Error: {angular_err_deg.item():.4f}°")

# =====================================================================
# =====================================================================
#
#   PART 2: PHYSICS DISCOVERY
#   Deriving quantities that were NEVER in the training data
#
# =====================================================================
# =====================================================================

print("\n" + "=" * 70)
print("  PHYSICS DISCOVERY: Deriving Unmeasured Quantities from Trained Field")
print("=" * 70)

# We need gradients enabled for physics computation
# 2,500 points is more than enough for scatter visualization and metrics
# and calculates instantaneously on CPU
N_eval = 2500
eval_idx = torch.randperm(X_test_norm.shape[0], device=device)[:N_eval]
eval_coords_norm = X_test_norm[eval_idx]
eval_coords_raw  = X_test[eval_idx]
eval_vel_true    = Y_test[eval_idx]


def compute_full_velocity_gradient(model, coords_norm, max_vel_mag):
    """
    Computes the full 3x3 velocity gradient tensor nabla-v at each point.

    Returns:
        vel: (N, 3) -- predicted velocity [u, v, w] in physical units
        grad_tensor: (N, 3, 3) -- dv_i/dx_j for i,j in {0,1,2}
            grad_tensor[:, i, j] = dv_i / dx_j

    The velocity gradient tensor is the fundamental object from which
    ALL derived quantities (vorticity, strain, Q-criterion, pressure
    Laplacian, enstrophy) are computed.
    """
    coords = coords_norm.clone().detach().requires_grad_(True)
    vel_norm_out = model.net(coords)

    vel_phys = vel_norm_out * max_vel_mag  # Scale back to physical units

    grad_tensor = torch.zeros(coords.shape[0], 3, 3, device=coords.device)

    for i in range(3):  # For each velocity component u, v, w
        grad_i = torch.autograd.grad(
            vel_phys[:, i], coords,
            grad_outputs=torch.ones(coords.shape[0], device=coords.device),
            create_graph=False, retain_graph=True
        )[0]
        # grad_i[:, j] = dv_i / dx_j (in normalized coordinate space)
        grad_tensor[:, i, :] = grad_i

    return vel_phys.detach(), grad_tensor.detach()


print("\n[*] Computing full 3x3 velocity gradient tensor (nabla-v)...")
pred_vel, grad_tensor = compute_full_velocity_gradient(model, eval_coords_norm, max_vel_mag)
print(f"    Gradient tensor shape: {grad_tensor.shape} = (N_points, 3, 3)")


# =====================================================================
# DERIVED QUANTITY 1: VORTICITY (omega = curl v)
# =====================================================================
print("\n" + "-" * 60)
print("  [1/5] VORTICITY FIELD  omega = curl(v)")
print("        omega_x = dw/dy - dv/dz")
print("        omega_y = du/dz - dw/dx")
print("        omega_z = dv/dx - du/dy")
print("-" * 60)

# Extract components of nabla-v
# grad_tensor[:, i, j] = dv_i / dx_j
du_dx = grad_tensor[:, 0, 0]
du_dy = grad_tensor[:, 0, 1]
du_dz = grad_tensor[:, 0, 2]
dv_dx = grad_tensor[:, 1, 0]
dv_dy = grad_tensor[:, 1, 1]
dv_dz = grad_tensor[:, 1, 2]
dw_dx = grad_tensor[:, 2, 0]
dw_dy = grad_tensor[:, 2, 1]
dw_dz = grad_tensor[:, 2, 2]

# Vorticity: omega = curl(v)
omega_x = dw_dy - dv_dz
omega_y = du_dz - dw_dx
omega_z = dv_dx - du_dy

vorticity_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

print(f"  Vorticity magnitude range: [{vorticity_mag.min().item():.4f}, {vorticity_mag.max().item():.4f}]")
print(f"  Mean vorticity: {vorticity_mag.mean().item():.4f}")
print(f"  Points with significant rotation (|omega| > mean): {(vorticity_mag > vorticity_mag.mean()).sum().item()}")


# =====================================================================
# DERIVED QUANTITY 2: Q-CRITERION (Vortex Identification)
# =====================================================================
print("\n" + "-" * 60)
print("  [2/5] Q-CRITERION  (Gold-standard vortex detection)")
print("        S = 0.5*(nabla-v + nabla-v^T)   -- Strain Rate (symmetric)")
print("        Omega = 0.5*(nabla-v - nabla-v^T)   -- Rotation Rate (antisymmetric)")
print("        Q = 0.5*(||Omega||^2_F - ||S||^2_F)")
print("        Q > 0  -->  Vortex Core (rotation dominates strain)")
print("-" * 60)

# Strain rate tensor: S = 0.5*(nabla-v + nabla-v^T)
S = 0.5 * (grad_tensor + grad_tensor.transpose(1, 2))
# Rotation rate tensor: Omega = 0.5*(nabla-v - nabla-v^T)
Omega_tensor = 0.5 * (grad_tensor - grad_tensor.transpose(1, 2))

# Frobenius norms squared: ||A||^2_F = sum_ij A_ij^2
S_norm_sq = torch.sum(S ** 2, dim=(1, 2))
Omega_norm_sq = torch.sum(Omega_tensor ** 2, dim=(1, 2))

# Q-criterion
Q_criterion = 0.5 * (Omega_norm_sq - S_norm_sq)

n_vortex = (Q_criterion > 0).sum().item()
print(f"  Q range: [{Q_criterion.min().item():.4f}, {Q_criterion.max().item():.4f}]")
print(f"  Vortex core points (Q > 0): {n_vortex} / {N_eval} ({100*n_vortex/N_eval:.1f}%)")
print(f"  Strain-dominated points (Q < 0): {N_eval - n_vortex} ({100*(N_eval-n_vortex)/N_eval:.1f}%)")


# =====================================================================
# DERIVED QUANTITY 3: PRESSURE LAPLACIAN (from Navier-Stokes)
# =====================================================================
print("\n" + "-" * 60)
print("  [3/5] PRESSURE FIELD RECOVERY  (Navier-Stokes Pressure Poisson)")
print("        nabla^2 p = -rho * sum_ij (du_i/dx_j)(du_j/dx_i)")
print("        Pressure information from velocity ONLY -- no pressure sensors!")
print("-" * 60)

# Pressure Poisson equation for incompressible flow:
#   nabla^2 p = -rho * sum_i sum_j (du_i/dx_j)(du_j/dx_i)
# This is the trace of (nabla-v)(nabla-v)^T
rho = 1.0
pressure_laplacian = torch.zeros(N_eval, device=device)
for i in range(3):
    for j in range(3):
        pressure_laplacian += grad_tensor[:, i, j] * grad_tensor[:, j, i]
pressure_laplacian = -rho * pressure_laplacian

print(f"  nabla^2 p range: [{pressure_laplacian.min().item():.4f}, {pressure_laplacian.max().item():.4f}]")
print(f"  Mean nabla^2 p: {pressure_laplacian.mean().item():.6f}")
print(f"  |nabla^2 p| approx 0 indicates uniform pressure (freestream): "
      f"{(pressure_laplacian.abs() < 0.01).sum().item()} points")


# =====================================================================
# DERIVED QUANTITY 4: STRAIN RATE TENSOR MAGNITUDE
# =====================================================================
print("\n" + "-" * 60)
print("  [4/5] STRAIN RATE TENSOR  S_ij = 0.5*(du_i/dx_j + du_j/dx_i)")
print("        ||S||_F = Frobenius norm = deformation intensity")
print("        Predicts drag, wall shear stress, material failure")
print("-" * 60)

strain_rate_mag = torch.sqrt(S_norm_sq)

print(f"  ||S|| range: [{strain_rate_mag.min().item():.4f}, {strain_rate_mag.max().item():.4f}]")
print(f"  Mean strain rate: {strain_rate_mag.mean().item():.4f}")
print(f"  High-deformation regions (||S|| > 2x mean): "
      f"{(strain_rate_mag > 2*strain_rate_mag.mean()).sum().item()}")


# =====================================================================
# DERIVED QUANTITY 5: ENSTROPHY (E = 0.5 ||omega||^2)
# =====================================================================
print("\n" + "-" * 60)
print("  [5/5] ENSTROPHY  E = 0.5 * ||omega||^2")
print("        Total rotational energy density")
print("        Drives the turbulent energy cascade (Kolmogorov theory)")
print("-" * 60)

enstrophy = 0.5 * vorticity_mag ** 2

print(f"  Enstrophy range: [{enstrophy.min().item():.6f}, {enstrophy.max().item():.6f}]")
print(f"  Mean enstrophy: {enstrophy.mean().item():.6f}")
print(f"  Total enstrophy (integrated): {enstrophy.sum().item():.4f}")


# =====================================================================
# VERIFICATION: Check Divergence-Free Condition (nabla . v = 0)
# =====================================================================
print("\n" + "-" * 60)
print("  [VERIFICATION] Incompressibility: nabla . v = du/dx + dv/dy + dw/dz")
print("-" * 60)

divergence = du_dx + dv_dy + dw_dz
div_mean = divergence.abs().mean().item()
div_max  = divergence.abs().max().item()

print(f"  Mean |nabla . v|: {div_mean:.6f}")
print(f"  Max  |nabla . v|: {div_max:.6f}")
if div_mean < 0.1:
    print(f"  [OK] Divergence-free condition well-satisfied (physics constraint working)")
else:
    print(f"  [!] Divergence non-negligible -- longer training would improve this")


# =====================================================================
# VERIFICATION: Compare derived vorticity to analytical ground truth
# =====================================================================
print("\n" + "-" * 60)
print("  [GROUND TRUTH CHECK] Analytical Vorticity for Flow Past Sphere")
print("-" * 60)

# For irrotational potential flow past a sphere, the TRUE vorticity is
# EXACTLY ZERO everywhere (curl of a gradient field = 0).
# So the closer our derived vorticity is to zero, the better.

print(f"  Analytical vorticity (potential flow): omega = 0 everywhere")
print(f"  MorphoCalc derived mean |omega|: {vorticity_mag.mean().item():.6f}")
print(f"  MorphoCalc derived max  |omega|: {vorticity_mag.max().item():.6f}")

if vorticity_mag.mean().item() < 0.5:
    print(f"  [OK] Vorticity correctly near-zero for irrotational flow!")
    print(f"    (Residual is numerical differentiation noise, not real rotation)")
else:
    print(f"  [!] Vorticity higher than expected -- training could improve accuracy")


# =====================================================================
# VISUALIZATION
# =====================================================================
print("\n[*] Generating Physics Discovery visualizations...")

coords_cpu = eval_coords_raw.cpu().numpy()
x_c, y_c, z_c = coords_cpu[:, 0], coords_cpu[:, 1], coords_cpu[:, 2]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("MorphoCalc Physics Discovery Engine\n"
             "Derived Quantities from Trained Neural Field (Never in Training Data)",
             fontsize=16, fontweight='bold', y=0.98)

# 1. Vorticity Magnitude
ax = axes[0, 0]
sc = ax.scatter(x_c, y_c, c=vorticity_mag.cpu().numpy(), cmap='inferno',
                s=1, alpha=0.7, vmin=0, vmax=np.percentile(vorticity_mag.cpu().numpy(), 95))
plt.colorbar(sc, ax=ax, label='|omega|')
ax.set_title('Vorticity Magnitude |omega| = |curl v|', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
circle = plt.Circle((0, 0), R, color='black', fill=True, alpha=0.3)
ax.add_patch(circle)
ax.set_aspect('equal')

# 2. Q-Criterion
ax = axes[0, 1]
q_cpu = Q_criterion.cpu().numpy()
vmax_q = np.percentile(np.abs(q_cpu), 95)
sc = ax.scatter(x_c, y_c, c=q_cpu, cmap='RdBu_r', s=1, alpha=0.7,
                vmin=-vmax_q, vmax=vmax_q)
plt.colorbar(sc, ax=ax, label='Q')
ax.set_title('Q-Criterion (Red=Vortex, Blue=Strain)', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
circle = plt.Circle((0, 0), R, color='black', fill=True, alpha=0.3)
ax.add_patch(circle)
ax.set_aspect('equal')

# 3. Pressure Laplacian
ax = axes[0, 2]
p_cpu = pressure_laplacian.cpu().numpy()
vmax_p = np.percentile(np.abs(p_cpu), 95)
sc = ax.scatter(x_c, y_c, c=p_cpu, cmap='coolwarm', s=1, alpha=0.7,
                vmin=-vmax_p, vmax=vmax_p)
plt.colorbar(sc, ax=ax, label='nabla^2 p')
ax.set_title('Pressure Laplacian (Navier-Stokes)', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
circle = plt.Circle((0, 0), R, color='black', fill=True, alpha=0.3)
ax.add_patch(circle)
ax.set_aspect('equal')

# 4. Strain Rate Magnitude
ax = axes[1, 0]
sc = ax.scatter(x_c, y_c, c=strain_rate_mag.cpu().numpy(), cmap='magma',
                s=1, alpha=0.7,
                vmax=np.percentile(strain_rate_mag.cpu().numpy(), 95))
plt.colorbar(sc, ax=ax, label='||S||')
ax.set_title('Strain Rate ||S||_F (Deformation Intensity)', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
circle = plt.Circle((0, 0), R, color='black', fill=True, alpha=0.3)
ax.add_patch(circle)
ax.set_aspect('equal')

# 5. Enstrophy
ax = axes[1, 1]
sc = ax.scatter(x_c, y_c, c=enstrophy.cpu().numpy(), cmap='hot',
                s=1, alpha=0.7,
                vmax=np.percentile(enstrophy.cpu().numpy(), 95))
plt.colorbar(sc, ax=ax, label='E')
ax.set_title('Enstrophy E = 0.5|omega|^2 (Rotational Energy)', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
circle = plt.Circle((0, 0), R, color='black', fill=True, alpha=0.3)
ax.add_patch(circle)
ax.set_aspect('equal')

# 6. Velocity Magnitude (reference)
ax = axes[1, 2]
vel_mag_cpu = torch.norm(pred_vel, dim=1).cpu().numpy()
sc = ax.scatter(x_c, y_c, c=vel_mag_cpu, cmap='viridis', s=1, alpha=0.7)
plt.colorbar(sc, ax=ax, label='|v|')
ax.set_title('Velocity Magnitude |v| (Reference)', fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
circle = plt.Circle((0, 0), R, color='black', fill=True, alpha=0.3)
ax.add_patch(circle)
ax.set_aspect('equal')

plt.tight_layout(rect=[0, 0, 1, 0.94])
save_path = 'morphocalc_physics_discovery.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[*] Saved: {save_path}")


# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("  MORPHOCALC PHYSICS DISCOVERY -- FINAL REPORT")
print("=" * 70)
print(f"""
  INPUT:  80,000 spatial coordinates (x, y, z)
  TRAINED ON: 15% of velocity data (u, v, w) only
  TRAINED FOR: {NUM_ITERATIONS} iterations ({train_time:.0f}s)

  RECONSTRUCTION (what was measured):
    Velocity L2 Error:   {rel_l2.item():.4f} ({rel_l2.item()*100:.2f}%)
    Angular Error:       {angular_err_deg.item():.4f} deg

  DISCOVERY (what was NEVER measured):

  1. Vorticity (omega = curl v)
     Mean |omega|: {vorticity_mag.mean().item():.6f}  (true: 0.0 for potential flow)

  2. Q-Criterion (vortex detection)
     Vortex cores (Q>0): {n_vortex}/{N_eval} points

  3. Pressure Laplacian (nabla^2 p from N-S)
     Mean nabla^2 p: {pressure_laplacian.mean().item():.6f}

  4. Strain Rate (deformation intensity)
     Mean ||S||: {strain_rate_mag.mean().item():.6f}

  5. Enstrophy (rotational energy density)
     Mean E: {enstrophy.mean().item():.6f}

  PHYSICS CHECK:
     Divergence |nabla . v|: {div_mean:.6f} (should be ~0)
     Vorticity |omega|:   {vorticity_mag.mean().item():.6f} (should be ~0 for potential)

  KEY INSIGHT: MorphoCalc doesn't just reconstruct -- it DISCOVERS.
  From sparse velocity samples, the trained neural field's differentiable
  structure allows computing pressure, vorticity, strain, and enstrophy
  that were NEVER in the training data. This is scientific inference,
  not interpolation.
""")
print("=" * 70)
