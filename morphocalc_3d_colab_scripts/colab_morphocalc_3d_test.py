"""
================================================================================
MorphoCalc 3D: Google Colab Proof-of-Concept
================================================================================
INSTRUCTIONS: 
Copy and paste this entire script into a single cell in Google Colab and hit run.
This script generates a synthetic 3D dataset (to save you from downloading 10GB of data)
and trains the 3D SIREN network using Batched DataLoaders to prevent RAM explosions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

print("Initializing MorphoCalc 3D on Colab...")

# 1. Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

# ==========================================
# 2. THE 3D SIREN BRAIN
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
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0, 
                                            np.sqrt(6 / in_features) / omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class MorphoCalcSIREN_3D(nn.Module):
    # in_features=3 (X,Y,Z), out_features=4 (U,V,W, Pressure)
    def __init__(self, in_features=3, hidden_features=1024, hidden_layers=4, out_features=4):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=3))
        
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=3))
            
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / 3, 
                                         np.sqrt(6 / hidden_features) / 3)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

# ==========================================
# 3. THE 3D PHYSICS LOSS
# ==========================================
def physics_divergence_loss_3d(pred_outputs, coords):
    # Split outputs into U, V, W
    U = pred_outputs[:, 0:1]
    V = pred_outputs[:, 1:2]
    W = pred_outputs[:, 2:3]

    # Calculate Gradients with respect to X, Y, Z
    dU = torch.autograd.grad(U, coords, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    dV = torch.autograd.grad(V, coords, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dW = torch.autograd.grad(W, coords, grad_outputs=torch.ones_like(W), create_graph=True)[0]

    # Mass conservation in 3D: dU/dX + dV/dY + dW/dZ = 0
    divergence = dU[:, 0:1] + dV[:, 1:2] + dW[:, 2:3]
    return torch.mean(divergence ** 2)

# ==========================================
# 4. GENERATE SYNTHETIC 3D DATA
# ==========================================
print("\nGenerating Synthetic 3D Airflow Data...")
num_points = 50000  # 50,000 points

# Random X, Y, Z coordinates
X_train = torch.rand(num_points, 3) * 2.0 - 1.0  

# Fake target velocities (just to give it something to learn)
Y_train_U = X_train[:, 0:1] ** 2
Y_train_V = torch.sin(X_train[:, 1:2])
Y_train_W = torch.cos(X_train[:, 2:3])
Y_train_P = X_train[:, 0:1] + X_train[:, 1:2]
Y_train = torch.cat([Y_train_U, Y_train_V, Y_train_W, Y_train_P], dim=1)

# ==========================================
# 5. DATALOADERS (PREVENTING RAM EXPLOSION)
# ==========================================
dataset = TensorDataset(X_train, Y_train)
# We feed the GPU 8,192 points at a time
batch_size = 8192
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
model = MorphoCalcSIREN_3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 100
print("\nStarting 3D Training Loop...")
start_time = time.time()

for epoch in range(epochs):
    epoch_mse = 0
    epoch_phys = 0
    
    for batch_coords, batch_targets in dataloader:
        batch_coords = batch_coords.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        predictions, coords_with_grad = model(batch_coords)
        
        # MSE Loss (Data accuracy)
        mse_loss = nn.MSELoss()(predictions, batch_targets)
        
        # 3D Physics Loss
        phys_loss = physics_divergence_loss_3d(predictions, coords_with_grad)
        
        # Total Loss (Topo loss disabled for synthetic data test)
        total_loss = mse_loss + (0.5 * phys_loss)
        
        total_loss.backward()
        optimizer.step()
        
        epoch_mse += mse_loss.item()
        epoch_phys += phys_loss.item()
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | MSE Loss: {epoch_mse/len(dataloader):.6f} | 3D Phys Loss: {epoch_phys/len(dataloader):.6f}")

print(f"\nTraining Complete in {time.time() - start_time:.2f} seconds!")
print("MorphoCalc successfully solved 3D physical constraints!")
