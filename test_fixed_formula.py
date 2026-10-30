"""
=====================================================================================
  MORPHOCALC-NEXUS: UNIFIED ROBUST FORMULA REPAIR & BENCHMARK
  Fixing both the Ringing/Crash (Pratyaksh-II) and Asymptotic Freezing (Quartic)
=====================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import numpy as np

def vdp1000_deriv(y):
    return np.array([y[1], 1000.0 * (1.0 - y[0]**2) * y[1] - y[0]])

# 1. Classical RK4
def step_rk4(y, dt, f):
    k1 = dt * f(y)
    k2 = dt * f(y + 0.5 * k1)
    k3 = dt * f(y + 0.5 * k2)
    k4 = dt * f(y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# 2. Pratyaksh-II (Old formulation)
def step_pratyaksh2(y, dt, f):
    k1 = dt * f(y)
    k2 = dt * f(y + 0.5 * k1)
    k3 = dt * f(y + 0.5 * k2)
    k4 = dt * f(y + k3)
    C = k4 - k3 - k2 + k1
    Sn = np.linalg.norm(y) + np.linalg.norm(k1) + 1e-12
    D = (0.1 * np.linalg.norm(C) / Sn)**2
    den = 6.0 + D
    return y + (k1 + 2*k2 + 2*k3 + k4) / den

# 3. MorphoCalc Quartic (Unbounded - causes freezing at huge dt)
def step_morphocalc_quartic(y, dt, f):
    k1 = dt * f(y)
    k2 = dt * f(y + 0.5 * k1)
    k3 = dt * f(y + 0.5 * k2)
    k4 = dt * f(y + k3)
    CA = k4 - 2*k3 + k2
    D = np.sum(CA**2) / (np.sum(k1**2) + 1e-14)
    den = 1.0 + 0.015 * (D**3) * (1.0 + D)
    return y + (k1 + 2*k2 + 2*k3 + k4) / (6.0 * den)

# 4. REPAIRED / BULLETPROOF FORMULA: MorphoCalc-Nexus Mark-III
# Combines:
# (a) Tangent-normalized curvature sensor: D = ||k4 - 2k3 + k2||^2 / (||k1||^2 + eps)
# (b) Rational Padé soft-saturating denominator: den = 1.0 + (0.05 * D^2) / (1.0 + 0.005 * D)
#     - Smooth limit D -> 0: den -> 1.000000 (Exact Order 4 RK4)
#     - Stiff regime D ~ 100: den -> smooth heavy damping without ringing
#     - Extreme regime D -> inf: den scales linearly ~ 10*D instead of exploding as D^4, avoiding freezing!
def step_morphocalc_repaired(y, dt, f):
    k1 = dt * f(y)
    k2 = dt * f(y + 0.5 * k1)
    k3 = dt * f(y + 0.5 * k2)
    k4 = dt * f(y + k3)
    CA = k4 - 2*k3 + k2
    # Dynamic noise guard for true equilibrium points
    norm_k1_sq = np.sum(k1**2)
    norm_ca_sq = np.sum(CA**2)
    if norm_k1_sq < 1e-20 and norm_ca_sq < 1e-20:
        D = 0.0
    else:
        D = norm_ca_sq / (norm_k1_sq + 1e-14)
    
    # Rational Padé soft-saturating denominator (Asymptotically regularized)
    D = min(float(D), 1e300)
    if D > 1e6:
        den = 1.0 + (10.0 * D) / (1.0 + 200.0 / D)
    else:
        den = 1.0 + (0.05 * (D**2)) / (1.0 + 0.005 * D)
    return y + (k1 + 2*k2 + 2*k3 + k4) / (6.0 * den)


print("=" * 80)
print("  TESTING REPAIRED FORMULA ON STIFF VAN DER POL (mu = 1000)")
print("=" * 80)

# Reference at t=0.5s via high-precision substepping
t_target = 0.5
n_ref = 100000
dt_ref = t_target / n_ref
y_ref = np.array([2.0, 0.0])
for _ in range(n_ref):
    k1 = dt_ref * vdp1000_deriv(y_ref)
    k2 = dt_ref * vdp1000_deriv(y_ref + 0.5 * k1)
    k3 = dt_ref * vdp1000_deriv(y_ref + 0.5 * k2)
    k4 = dt_ref * vdp1000_deriv(y_ref + k4 if False else y_ref + k3)
    y_ref += (k1 + 2*k2 + 2*k3 + k4) / 6.0

print(f"Reference Truth at t = {t_target}s: y = [{y_ref[0]:.6f}, {y_ref[1]:.6f}]\n")

print(f"{'dt':<8} | {'RK4':<10} | {'Pratyaksh-II':<16} | {'Quartic (Old)':<16} | {'Repaired Mark-III':<16}")
print("-" * 75)

for dt in [0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200]:
    n_steps = int(round(t_target / dt))
    
    # 1. RK4
    y = np.array([2.0, 0.0])
    c1 = False
    for _ in range(n_steps):
        y = step_rk4(y, dt, vdp1000_deriv)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.max(np.abs(y)) > 1e6:
            c1 = True; break
    err1 = f"{np.linalg.norm(y - y_ref):.2e}" if not c1 else "CRASH (NaN)"
    
    # 2. Pratyaksh-II
    y = np.array([2.0, 0.0])
    c2 = False
    for _ in range(n_steps):
        y = step_pratyaksh2(y, dt, vdp1000_deriv)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.max(np.abs(y)) > 1e6:
            c2 = True; break
    err2 = f"{np.linalg.norm(y - y_ref):.2e}" if not c2 else "CRASH (NaN)"
    
    # 3. MorphoCalc Quartic (Old)
    y = np.array([2.0, 0.0])
    c3 = False
    for _ in range(n_steps):
        y = step_morphocalc_quartic(y, dt, vdp1000_deriv)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.max(np.abs(y)) > 1e6:
            c3 = True; break
    err3 = f"{np.linalg.norm(y - y_ref):.2e}" if not c3 else "CRASH (NaN)"
    
    # 4. Repaired Mark-III
    y = np.array([2.0, 0.0])
    c4 = False
    for _ in range(n_steps):
        y = step_morphocalc_repaired(y, dt, vdp1000_deriv)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.max(np.abs(y)) > 1e6:
            c4 = True; break
    err4 = f"{np.linalg.norm(y - y_ref):.2e}" if not c4 else "CRASH (NaN)"
    
    print(f"{dt:<8.3f} | {err1:<10} | {err2:<16} | {err3:<16} | {err4:<16}")

print("\n" + "=" * 80)
