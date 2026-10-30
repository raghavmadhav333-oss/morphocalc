# ==============================================================================
# MORPHOCALC UNIVERSAL - OVERNIGHT KAGGLE TRAINING SCRIPT
# ==============================================================================

!pip install torch_geometric

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.datasets import AirfRANS
import numpy as np

# 1. GPU Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. Download and Load Multiple Supercomputer Simulations
print("Loading the AirfRANS Dataset...")
dataset = AirfRANS(root='./AirfRANS_data', task='full', train=True)

X_list, Y_list, U_list, V_list, P_list = [], [], [], [], []

# We will load 100 unique wings to build the Universal Model.
# This yields about 15 MILLION physics points (Massive scale dataset)
NUM_SIMULATIONS = 100 
print(f"Extracting {NUM_SIMULATIONS} unique airplane wings for Universal Training...")

for i in range(NUM_SIMULATIONS):
    data = dataset[i]
    X_list.append(data.pos[:, 0].numpy())
    Y_list.append(data.pos[:, 1].numpy())
    U_list.append(data.y[:, 0].numpy())
    V_list.append(data.y[:, 1].numpy())
    P_list.append(data.y[:, 2].numpy())
    
    if i % 20 == 0:
        print(f"Processed {i}/{NUM_SIMULATIONS} wings...")

# Concatenate into massive arrays
X = np.concatenate(X_list)
Y = np.concatenate(Y_list)
U_real = np.concatenate(U_list)
V_real = np.concatenate(V_list)
P_real = np.concatenate(P_list)

print(f"\nTotal Physics Data Points for Training: {X.shape[0]:,}")

# Pad 3D dimensions
N = X.shape[0]
Z = np.zeros(N)
SDF = np.zeros(N)
Nx = np.zeros(N)
Ny = np.zeros(N)
Nz = np.zeros(N)
W_real = np.zeros(N)

# Build Tensors
inputs = np.column_stack([X, Y, Z, SDF, Nx, Ny, Nz])
targets = np.column_stack([U_real, V_real, W_real, P_real])

# Normalize
targets_mean = targets.mean(axis=0)
targets_std = targets.std(axis=0) + 1e-8
targets_normalized = (targets - targets_mean) / targets_std

X_train = torch.FloatTensor(inputs)
Y_train = torch.FloatTensor(targets_normalized)

# ==============================================================================
# 3. MORPHOCALC UNIVERSAL (DEEPER ARCHITECTURE)
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

class MorphoCalcUniversal(nn.Module):
    def __init__(self):
        super().__init__()
        # 🚀 Reduced to 256 to speed up physics gradients (autograd is heavy!)
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
# 4. OVERNIGHT TRAINING LOOP
# ==============================================================================
model = MorphoCalcUniversal().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Massive batch size to handle 15 million points efficiently
loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32768, shuffle=True)
mse_fn = nn.MSELoss()

print(f"\n🚀 Starting UNIVERSAL Training on {DEVICE}...")

# 🚀 Changed to 100 Epochs (100 epochs * 15M points = 1.5 Billion samples seen!)
TOTAL_EPOCHS = 100 
for epoch in range(TOTAL_EPOCHS):
    e_mse, e_phys = 0.0, 0.0
    for bx, by in loader:
        optimizer.zero_grad()
        
        bx = bx.to(DEVICE).requires_grad_(True)
        by = by.to(DEVICE)
        
        preds = model(bx)
        loss_mse = mse_fn(preds, by)
        
        # Physics Loss
        u, v, w = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
        dudx = torch.autograd.grad(u, bx, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
        dvdy = torch.autograd.grad(v, bx, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1:2]
        dwdz = torch.autograd.grad(w, bx, grad_outputs=torch.ones_like(w), create_graph=True)[0][:, 2:3]
        
        div = dudx + dvdy + dwdz
        loss_phys = torch.mean(div**2)
        
        loss = loss_mse + 0.1 * loss_phys
        loss.backward()
        optimizer.step()
        
        e_mse += loss_mse.item()
        e_phys += loss_phys.item()
        
    if epoch % 5 == 0:
        print(f"Epoch {epoch:04d}/{TOTAL_EPOCHS} | MSE: {e_mse/len(loader):.5f} | Physics: {e_phys/len(loader):.5f}")
        # 🚀 SAVE CHECKPOINT EVERY 5 EPOCHS SO WE NEVER LOSE DATA
        torch.save(model.state_dict(), f'morphocalc_checkpoint_epoch_{epoch}.pt')

print("\n✅ Universal Training Complete!")
torch.save(model.state_dict(), 'morphocalc_universal_airfrans.pt')
print("Weights saved to morphocalc_universal_airfrans.pt")
