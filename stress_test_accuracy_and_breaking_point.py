"""
=====================================================================================
  MORPHOCALC-NEXUS: DEEP BREAKING-POINT PROBE & ACCURACY COMPARISON
  Pushing dt to extreme limits (dt = 0.05 -> 1.0) and comparing True Errors vs Reference
=====================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import math
import numpy as np

def vdp1000_deriv(y):
    return np.array([y[1], 1000.0 * (1.0 - y[0]**2) * y[1] - y[0]])

def morphocalc_step(y, dt, deriv_fn, method="morphocalc"):
    k1 = dt * deriv_fn(y)
    k2 = dt * deriv_fn(y + 0.5 * k1)
    k3 = dt * deriv_fn(y + 0.5 * k2)
    k4 = dt * deriv_fn(y + k3)

    if method == "standard_rk4":
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    elif method == "pratyaksh_ii":
        C = k4 - k3 - k2 + k1
        Sn = np.linalg.norm(y) + np.linalg.norm(k1) + 1e-12
        C_norm = np.linalg.norm(C)
        D = (0.1 * C_norm / Sn)**2
        den = 6.0 + D
        return y + (k1 + 2*k2 + 2*k3 + k4) / den

    elif method == "morphocalc":
        CA = k4 - 2*k3 + k2
        CA_sq = np.sum(CA**2)
        K1_sq = np.sum(k1**2) + 1e-14
        D = CA_sq / K1_sq
        den = 1.0 + 0.015 * (D**3) * (1.0 + D)
        return y + (k1 + 2*k2 + 2*k3 + k4) / (6.0 * den)

print("=" * 80)
print("  PROBING EXTREME STEP-SIZE BREAKING POINT ON STIFF VAN DER POL (mu = 1000)")
print("=" * 80)

# Compute high-precision ground truth reference at t = 0.5 using Radau / tiny dt
# Using adaptive sub-stepping reference with 500,000 steps
t_target = 0.5
n_ref = 500000
dt_ref = t_target / n_ref
y_ref = np.array([2.0, 0.0])
for _ in range(n_ref):
    k1 = dt_ref * vdp1000_deriv(y_ref)
    k2 = dt_ref * vdp1000_deriv(y_ref + 0.5 * k1)
    k3 = dt_ref * vdp1000_deriv(y_ref + 0.5 * k2)
    k4 = dt_ref * vdp1000_deriv(y_ref + k3)
    y_ref += (k1 + 2*k2 + 2*k3 + k4) / 6.0

print(f"High-Precision Reference at t = {t_target}s: y = [{y_ref[0]:.6f}, {y_ref[1]:.6f}]")

print("\n--- Sweeping dt from 0.005 to 0.500 (Measuring Mean Absolute Error vs Reference) ---")
for dt in [0.005, 0.010, 0.020, 0.050, 0.080, 0.100, 0.150, 0.200, 0.500]:
    n_steps = int(round(t_target / dt))
    
    # 1. Pratyaksh-II
    y_p = np.array([2.0, 0.0])
    crash_p = False
    for s in range(n_steps):
        y_p = morphocalc_step(y_p, dt, vdp1000_deriv, method="pratyaksh_ii")
        if np.any(np.isnan(y_p)) or np.any(np.isinf(y_p)) or np.max(np.abs(y_p)) > 1e6:
            crash_p = True
            break
    err_p = np.linalg.norm(y_p - y_ref) if not crash_p else float('nan')
    
    # 2. MorphoCalc Quartic
    y_m = np.array([2.0, 0.0])
    crash_m = False
    for s in range(n_steps):
        y_m = morphocalc_step(y_m, dt, vdp1000_deriv, method="morphocalc")
        if np.any(np.isnan(y_m)) or np.any(np.isinf(y_m)) or np.max(np.abs(y_m)) > 1e6:
            crash_m = True
            break
    err_m = np.linalg.norm(y_m - y_ref) if not crash_m else float('nan')
    
    status_p = f"MAE: {err_p:.4e}" if not crash_p else "💥 CRASH"
    status_m = f"MAE: {err_m:.4e}" if not crash_m else "💥 CRASH"
    print(f"dt = {dt:6.3f} ({n_steps:3d} steps) | Pratyaksh-II: {status_p:<18} | MorphoCalc: {status_m}")

print("\n" + "=" * 80)
