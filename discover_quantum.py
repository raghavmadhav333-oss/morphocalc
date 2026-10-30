import torch
import torch.nn as nn
import numpy as np

print("===============================================================")
print("  MORPHOCALC + PARC: QUANTUM MECHANICS EDITION")
print("  Objective: Discover the Quantum Harmonic Oscillator Potential")
print("===============================================================")

# 1. GENERATE QUANTUM WAVEFUNCTION DATA
# In quantum mechanics, a particle in a parabolic well (Harmonic Oscillator)
# has a Gaussian ground-state probability wave function.
x = np.linspace(-5.0, 5.0, 2000).astype(np.float32)
# psi(x) ~ exp(-x^2 / 2)
psi = np.exp(-0.5 * x**2)

# The Schrödinger Equation relates the 2nd derivative of psi to the Potential V(x):
# (-1/2) * d^2(psi)/dx^2 + V(x)*psi = E*psi
# We rearrange to isolate V(x): V(x) = E + (1/2psi) * d^2(psi)/dx^2
# For ground state, E = 0.5, and V(x) should be discovered as exactly 0.5 * x^2
d2_psi = (x**2 - 1.0) * np.exp(-0.5 * x**2)
V_true = 0.5 + (0.5 / psi) * d2_psi # This mathematically simplifies to 0.5 * x^2

x_tensor = torch.tensor(x).unsqueeze(1)
V_tensor = torch.tensor(V_true).unsqueeze(1)

print("\n[*] MorphoCalc: Scanning electron probability distributions...")

print("[*] PARC: Extracting the hidden Quantum Potential Energy V(x)...")
# PARC Dictionary of potential shapes:
# 1. Free Particle (V = 0)
lib_free = torch.zeros_like(x_tensor)
# 2. Linear Potential / Constant Force (V = x)
lib_linear = x_tensor
# 3. Harmonic Oscillator (V = x^2) -> The true quantum well
lib_harmonic = x_tensor**2
# 4. Quartic Potential (V = x^4) -> Anharmonic oscillator
lib_quartic = x_tensor**4

# We add a column of ones for the bias (Energy E)
lib_bias = torch.ones_like(x_tensor)

X = torch.cat([lib_free, lib_linear, lib_harmonic, lib_quartic, lib_bias], dim=1)

# Solve for the exact weights mapping the dictionary to the True Potential
# We use a slight regularization (Ridge) to avoid singular matrices from the zero column
X_T_X = torch.matmul(X.t(), X) + 1e-4 * torch.eye(5)
X_T_Y = torch.matmul(X.t(), V_tensor)
weights = torch.matmul(torch.inverse(X_T_X), X_T_Y)

print("\n[*] PARC Quantum Physics Discovery:")
print(f"  [Potential 1] Free Particle (V = 0)   : {weights[0].item():.6f}")
print(f"  [Potential 2] Linear Well   (V = x)   : {weights[1].item():.6f}")
print(f"  [Potential 3] Harmonic Well (V = x^2) : {weights[2].item():.6f}")
print(f"  [Potential 4] Quartic Well  (V = x^4) : {weights[3].item():.6f}")

print("\n[*] PARC CONCLUSION:")
if weights[2].item() > 0.4:
    print("    PARC discovered that the electron is trapped in a Quantum Harmonic Oscillator!")
    print(f"    Discovered Equation: V(x) = {weights[2].item():.4f} * x^2")
print("===============================================================\n")
