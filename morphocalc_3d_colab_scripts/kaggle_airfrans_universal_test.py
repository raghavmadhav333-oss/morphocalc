# ==============================================================================
# MORPHOCALC UNIVERSAL - UNSEEN WING TESTING SCRIPT
# ==============================================================================

!pip install torch_geometric

import torch
import torch.nn as nn
from torch_geometric.datasets import AirfRANS
import numpy as np
import matplotlib.pyplot as plt

# 1. DEFINE THE ARCHITECTURE (MUST MATCH THE UNIVERSAL TRAINING SCRIPT)
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
        self.net = nn.Sequential(
            SineLayer(7, 512, is_first=True),
            SineLayer(512, 512),
            SineLayer(512, 512),
            SineLayer(512, 512),
            SineLayer(512, 512)
        )
        self.final = nn.Linear(512, 4)

    def forward(self, x):
        return self.final(self.net(x))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. LOAD THE TRAINED BRAIN
# ==============================================================================
# 🛑 IMPORTANT: Update this path to wherever your .pt file is saved in Kaggle!
WEIGHTS_PATH = '/kaggle/input/YOUR_DATASET_OR_OUTPUT/morphocalc_universal_airfrans.pt' 

try:
    model = MorphoCalcUniversal().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Universal Brain Loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: Make sure you put the correct file path!\n{e}")


# ==============================================================================
# 3. GET AN UNSEEN WING (Wing #101)
# ==============================================================================
print("Loading dataset to find an Unseen Wing...")
dataset = AirfRANS(root='./AirfRANS_data', task='full', train=True)

# The AI only trained on Wings 0 to 99. It has NEVER seen Wing 101!
unseen_data = dataset[101] 

X = unseen_data.pos[:, 0].numpy()
Y = unseen_data.pos[:, 1].numpy()
U_real = unseen_data.y[:, 0].numpy()

N = X.shape[0]
inputs = np.column_stack([X, Y, np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)])
X_test = torch.FloatTensor(inputs).to(DEVICE)

# ==============================================================================
# 4. PREDICT & VISUALIZE (Research-READY BOARD GRAPHICS)
# ==============================================================================
print("Predicting fluid dynamics for Unseen Wing #101...")
with torch.no_grad():
    preds = model(X_test).cpu().numpy()

skip = 5 # Plot every 5th point for speed
X_fast = X[::skip]
Y_fast = Y[::skip]

# We must un-normalize the AI's prediction to match the real-world scale
local_mean = U_real.mean()
local_std = U_real.std() + 1e-8
U_pred_raw = (preds[::skip, 0] * local_std) + local_mean 
U_real_raw = U_real[::skip]

# Calculate absolute error and MAE
error = np.abs(U_real_raw - U_pred_raw)
mae = np.mean(error)

# Lock scales so colors match perfectly
vmin = min(U_real_raw.min(), U_pred_raw.min())
vmax = max(U_real_raw.max(), U_pred_raw.max())

plt.figure(figsize=(20, 6))
plt.suptitle(f'Universal Model Generalization on UNSEEN Wing #101\nMean Absolute Error (MAE): {mae:.4f}', fontsize=16)

# Plot 1: REAL
plt.subplot(1, 3, 1)
plt.scatter(X_fast, Y_fast, c=U_real_raw, cmap='jet', s=2, vmin=vmin, vmax=vmax)
plt.colorbar(label='Wind Velocity (U)')
plt.title('REAL Supercomputer Data')
plt.axis('equal')

# Plot 2: AI PREDICTION
plt.subplot(1, 3, 2)
plt.scatter(X_fast, Y_fast, c=U_pred_raw, cmap='jet', s=2, vmin=vmin, vmax=vmax)
plt.colorbar(label='Wind Velocity (U)')
plt.title('MorphoCalc AI Prediction')
plt.axis('equal')

# Plot 3: ERROR DIFFERENTIAL
plt.subplot(1, 3, 3)
plt.scatter(X_fast, Y_fast, c=error, cmap='Reds', s=2, vmin=0, vmax=vmax*0.3) 
plt.colorbar(label='Absolute Error')
plt.title('Differential Plot (Error)')
plt.axis('equal')

plt.tight_layout()
plt.show()
