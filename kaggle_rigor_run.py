import torch
import torch.nn as nn
import numpy as np
import urllib.request
import os

# ==========================================
# 1. DATA GENERATION / DOWNLOAD MODULE
# ==========================================
def get_cfd_data():
    """
    Since local storage and internet are limited, we generate a synthetic 
    noisy 2D fluid wake directly in the Kaggle RAM to test the SINDy closure.
    """
    print("Generating Synthetic 2D Wake Data (In-Memory)...")
    np.random.seed(42)
    
    # Grid
    x = np.linspace(0, 10, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Synthetic Base Flow (e.g. flow past a cylinder approximation)
    u_base = 1.0 - np.exp(-(Y**2)) 
    v_base = 0.1 * np.sin(X) * np.exp(-(Y**2))
    
    # Add synthetic "turbulence" (high frequency noise)
    noise_level = 0.15 # 15% relative noise (High Noise!)
    u_noisy = u_base + noise_level * np.random.randn(*u_base.shape)
    v_noisy = v_base + noise_level * np.random.randn(*v_base.shape)
    
    # Out of Distribution (OOD) Data - Flat plate boundary layer profile
    u_ood = Y**(1/7.0) # 1/7th power law
    u_ood = np.nan_to_num(u_ood, nan=0.0) # clean negatives
    u_ood_noisy = u_ood + noise_level * np.random.randn(*u_ood.shape)
    
    print("Data generation complete. Noise level: 15%")
    return (X, Y, u_noisy, v_noisy), (X, Y, u_ood_noisy, np.zeros_like(v_noisy))


# ==========================================
# 2. MORPHOCALC (SIREN) SMOOTHING MODULE
# ==========================================
class SIRENSmoother(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(), # Using Tanh for stable derivatives instead of raw SIREN sin() to avoid high-freq noise fitting
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2) # Predicts [u, v]
        )
    def forward(self, xy):
        return self.net(xy)

def train_smoother(X, Y, u, v):
    print("Training MorphoCalc Continuous Field to smooth spatial noise...")
    model = SIRENSmoother().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Flatten data
    xy = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32).cuda()
    uv_target = torch.tensor(np.stack([u.flatten(), v.flatten()], axis=1), dtype=torch.float32).cuda()
    
    for epoch in range(2000):
        pred = model(xy)
        loss = nn.MSELoss()(pred, uv_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Smoothing Epoch {epoch} | Loss: {loss.item():.4f}")
            
    return model

# ==========================================
# 3. KAGGLE EXECUTION ORCHESTRATION
# ==========================================
if __name__ == "__main__":
    print("=== MORPHOCALC KAGGLE RIGOR RUN ===")
    
    # 1. Get Data
    train_data, ood_data = get_cfd_data()
    X, Y, u_noisy, v_noisy = train_data
    
    # 2. Smooth the extremely noisy data (15% noise) using Neural Field
    # This directly addresses the SINDy derivative bottleneck.
    smoother = train_smoother(X, Y, u_noisy, v_noisy)
    
    print("\nPhase 1: Smoothing Complete. Ready for SINDy Extraction.")
    print("Next Step on Kaggle: Pass the cleaned `smoother` outputs to the PARC SINDy Library.")
    print("Baseline Comparison and OOD checks will be executed against the synthetic truth.")
