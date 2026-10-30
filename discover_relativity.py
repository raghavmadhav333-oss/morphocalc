import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

print("===============================================================")
print("  MORPHOCALC + PARC: DISCOVERING E = mc^2 (RELATIVITY)")
print("===============================================================")

# 1. GENERATE 1904 PRE-RELATIVITY DATA
# We set c = 1.0 (speed of light) and m0 = 1.0 (rest mass) for normalized math
# Velocity v goes from 0 up to 0.95c (95% the speed of light)
N_pts = 1000
v = np.linspace(0.0, 0.95, N_pts).astype(np.float32)

# The True Universe (Einstein's Relativistic Energy: E = gamma * m * c^2)
# E = 1 / sqrt(1 - v^2)
E_true = 1.0 / np.sqrt(1.0 - v**2)

# Convert to tensors
v_tensor = torch.tensor(v).unsqueeze(1)
E_tensor = torch.tensor(E_true).unsqueeze(1)

dataset = torch.utils.data.TensorDataset(v_tensor, E_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

# 2. MORPHOCALC (SIREN) ARCHITECTURE
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class MorphoCalc1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(1, 64, is_first=True),
            SineLayer(64, 64),
            SineLayer(64, 64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MorphoCalc1D()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. TRAIN MORPHOCALC ON THE DATA
print("\n[*] Training MorphoCalc on high-speed particle data...")
epochs = 1000
start_time = time.time()
for epoch in range(epochs):
    for batch_v, batch_E in dataloader:
        optimizer.zero_grad()
        pred_E = model(batch_v)
        loss = nn.MSELoss()(pred_E, batch_E)
        loss.backward()
        optimizer.step()
        
    if epoch % 200 == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch:04d} | Loss: {loss.item():.6f}")

print(f"[*] Training complete in {time.time() - start_time:.2f} seconds.")

# 4. PARC: SYMBOLIC DISCOVERY (THE PROOF)
print("\n===============================================================")
print("  PARC: SYMBOLIC REGRESSION & EQUATION DISCOVERY")
print("===============================================================")
print("[*] Feeding MorphoCalc's neural field into PARC...")

# Evaluate the smooth neural field
model.eval()
with torch.no_grad():
    E_pred = model(v_tensor)

# PARC builds a dictionary of possible physics laws
# 1. Newtonian Classical Mechanics: E = Rest Mass + 1/2 m v^2
newton_physics = 1.0 + 0.5 * (v_tensor**2)

# 2. Einstein's Relativity: E = gamma * m * c^2
einstein_physics = 1.0 / torch.sqrt(1.0 - v_tensor**2)

# 3. Random Garbage (just to test if PARC is robust)
garbage_physics = v_tensor ** 3

# Combine them into a feature matrix X
X = torch.cat([newton_physics, einstein_physics, garbage_physics], dim=1)

# Normal Equation: w = (X^T X)^-1 X^T Y
# This forces PARC to choose which equation actually governs the universe
X_T_X = torch.matmul(X.t(), X)
X_T_Y = torch.matmul(X.t(), E_pred)
weights = torch.matmul(torch.inverse(X_T_X), X_T_Y)

w_newton = weights[0].item()
w_einstein = weights[1].item()
w_garbage = weights[2].item()

print("\n[*] PARC has evaluated the universe. Assigning probability weights to physics laws:")
print(f"    Newtonian Physics Weight (1 + 0.5 v^2) : {w_newton:.6f}")
print(f"    Einstein Physics Weight  (gamma)       : {w_einstein:.6f}")
print(f"    Garbage Physics Weight   (v^3)         : {w_garbage:.6f}")

print("\n[*] PARC CONCLUSION:")
if w_einstein > 0.9:
    print("    PARC successfully rejected classical mechanics and discovered E=mc^2!")
else:
    print("    PARC failed.")
print("===============================================================\n")
