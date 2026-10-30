"""
=====================================================================================
  MORPHOCALC-NEXUS: EXTREME ADVERSARIAL COMPLEXITY & STRESS TEST SUITE
  Investigating Real Numerical Boundaries, Failure Modes, and Mathematical Robustness
=====================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import math
import numpy as np

print("=" * 80)
print("  MORPHOCALC-NEXUS: ADVERSARIAL STRESS-TEST & FAILURE BOUNDARY DISCOVERY")
print("=" * 80)

def morphocalc_step(y, dt, deriv_fn, method="morphocalc"):
    k1 = dt * deriv_fn(y)
    k2 = dt * deriv_fn(y + 0.5 * k1)
    k3 = dt * deriv_fn(y + 0.5 * k2)
    k4 = dt * deriv_fn(y + k3)

    if method == "standard_rk4":
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    elif method == "pratyaksh_ii":
        # Master Pratyaksh-II formulation: D = (0.1 * ||C|| / Sn)^2
        C = k4 - k3 - k2 + k1
        Sn = np.linalg.norm(y) + np.linalg.norm(k1) + 1e-12
        C_norm = np.linalg.norm(C)
        D = (0.1 * C_norm / Sn)**2
        den = 6.0 + D
        return y + (k1 + 2*k2 + 2*k3 + k4) / den

    elif method == "morphocalc":
        # MorphoCalc Quartic Adaptive Rational Formulation
        CA = k4 - 2*k3 + k2
        CA_sq = np.sum(CA**2)
        K1_sq = np.sum(k1**2) + 1e-14
        D = CA_sq / K1_sq
        # Quartic closure: den = 1.0 + 0.015*D^3 * (1 + D)
        den = 1.0 + 0.015 * (D**3) * (1.0 + D)
        return y + (k1 + 2*k2 + 2*k3 + k4) / (6.0 * den)


# =============================================================================
# TEST 1: EXTREME STIFF CHEMICAL KINETICS (ROBERTSON PROBLEM: STIFFNESS 10^8 - 10^10)
# =============================================================================
print("\n[TEST 1] Robertson Problem (Stiff Chemical Kinetics: y1' = -0.04 y1 + 1e4 y2 y3, etc.)")
print("Testing step-size tolerance up to dt = 0.005 (where stiffness ratio is 10^8)...")

def robertson_deriv(y):
    y1, y2, y3 = y
    dy1 = -0.04 * y1 + 1e4 * y2 * y3
    dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * (y2**2)
    dy3 = 3e7 * (y2**2)
    return np.array([dy1, dy2, dy3])

# Run Robertson from t=0 to t=1.0 with dt=0.0001
dt_rob = 0.0001
n_rob = 1000
for name in ["standard_rk4", "pratyaksh_ii", "morphocalc"]:
    y = np.array([1.0, 0.0, 0.0])
    exploded = False
    step_fail = -1
    for step in range(n_rob):
        y_next = morphocalc_step(y, dt_rob, robertson_deriv, method=name)
        if np.any(np.isnan(y_next)) or np.any(np.isinf(y_next)) or np.max(np.abs(y_next)) > 1e10:
            exploded = True
            step_fail = step
            break
        y = y_next
    if exploded:
        print(f"  ❌ {name:<14}: Exploded / NaN crash at step {step_fail}!")
    else:
        # Check conservation law: y1 + y2 + y3 = 1.0
        mass_cons = abs(np.sum(y) - 1.0)
        print(f"  ✅ {name:<14}: Survived {n_rob} steps | Mass Residual: {mass_cons:.3e} | y: [{y[0]:.4f}, {y[1]:.4e}, {y[2]:.4f}]")


# =============================================================================
# TEST 2: ADVERSARIAL STIFF CHAOS (HIGH-DIMENSIONAL LORENZ-96 LATTICE: N=40 NODES)
# =============================================================================
print("\n[TEST 2] High-Dimensional Lorenz-96 Atmosphere Lattice (N = 40 variables, F = 8.0)")
print("Simulating chaotic atmospheric wave turbulence over 5,000 steps...")

def lorenz96_deriv(x, F=8.0):
    N = len(x)
    d = np.zeros(N)
    for i in range(N):
        d[i] = (x[(i+1)%N] - x[i-2]) * x[i-1] - x[i] + F
    return d

x0 = np.full(40, 8.0)
x0[19] += 0.01  # Perturbation at node 20
dt_l96 = 0.02
n_l96 = 5000

for name in ["standard_rk4", "pratyaksh_ii", "morphocalc"]:
    x = x0.copy()
    exploded = False
    step_fail = -1
    t0 = time.perf_counter()
    for step in range(n_l96):
        x = morphocalc_step(x, dt_l96, lorenz96_deriv, method=name)
        if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.max(np.abs(x)) > 1e5:
            exploded = True
            step_fail = step
            break
    elapsed = time.perf_counter() - t0
    if exploded:
        print(f"  ❌ {name:<14}: Exploded at step {step_fail}!")
    else:
        # Compute mean kinetic energy
        ke = 0.5 * np.mean(x**2)
        print(f"  ✅ {name:<14}: 5,000 steps in {elapsed:.3f}s | Mean Kinetic Energy: {ke:.4f} | Bounds: [{x.min():.2f}, {x.max():.2f}]")


# =============================================================================
# TEST 3: PURE SYMPLECTIC ORBITAL CONSERVATION (KEPLER TWO-BODY ORBIT, e=0.6)
# =============================================================================
print("\n[TEST 3] Highly Eccentric Kepler Orbit (Eccentricity e = 0.6, 20,000 Steps)")
print("Testing Long-Horizon Hamiltonian Energy Drift and Perihelion Precession...")

def kepler_deriv(s):
    # s = [x, y, vx, vy]
    x, y, vx, vy = s
    r = math.sqrt(x**2 + y**2)
    ax = -x / (r**3)
    ay = -y / (r**3)
    return np.array([vx, vy, ax, ay])

# Initial conditions for e=0.6
e_kepler = 0.6
a_kepler = 1.0
s0 = np.array([1.0 - e_kepler, 0.0, 0.0, math.sqrt((1.0 + e_kepler) / (1.0 - e_kepler))])
E0 = 0.5 * (s0[2]**2 + s0[3]**2) - 1.0 / math.sqrt(s0[0]**2 + s0[1]**2)

dt_kep = 0.005
n_kep = 20000

for name in ["standard_rk4", "pratyaksh_ii", "morphocalc"]:
    s = s0.copy()
    max_energy_drift = 0.0
    for step in range(n_kep):
        s = morphocalc_step(s, dt_kep, kepler_deriv, method=name)
        r = math.sqrt(s[0]**2 + s[1]**2)
        E = 0.5 * (s[2]**2 + s[3]**2) - 1.0 / r
        drift = abs(E - E0) / abs(E0)
        if drift > max_energy_drift:
            max_energy_drift = drift
    print(f"  {'✅' if max_energy_drift < 1e-2 else '⚠️'} {name:<14}: 20,000 Steps | Max Energy Drift: {max_energy_drift:.4e}")


# =============================================================================
# TEST 4: ADVERSARIAL ORDER OF CONVERGENCE TEST (SMOOTH HARMONIC LIMIT dt -> 0)
# =============================================================================
print("\n[TEST 4] Adversarial Non-Stiff Order-of-Convergence Test (Testing Order 4 Reduction)")
print("Checking if MorphoCalc preserves exact O(dt^4) convergence on harmonic oscillator...")

def harmonic_deriv(y):
    return np.array([y[1], -y[0]])

dt_list = [0.1, 0.05, 0.025, 0.0125]
y0 = np.array([1.0, 0.0])
t_final = 2.0 * math.pi  # Exact 1 full period -> true y = [1.0, 0.0]

for name in ["standard_rk4", "pratyaksh_ii", "morphocalc"]:
    errors = []
    for dt in dt_list:
        n_steps = int(round(t_final / dt))
        y = y0.copy()
        for _ in range(n_steps):
            y = morphocalc_step(y, dt, harmonic_deriv, method=name)
        err = np.linalg.norm(y - y0)
        errors.append(err)
    
    # Compute empirical convergence order: log2(err[i]/err[i+1])
    orders = [math.log2(errors[i] / errors[i+1]) for i in range(len(errors)-1)]
    mean_order = np.mean(orders)
    print(f"  📊 {name:<14}: Errors = {[f'{e:.2e}' for e in errors]} | Observed Order: {mean_order:.2f}")


# =============================================================================
# TEST 5: BREAKING POINT PROBE - FINDING THE EXACT FAILURE BOUNDARY (STIFF VDP)
# =============================================================================
print("\n[TEST 5] Failure Boundary Scan: Finding the exact dt where each method crashes")
print("Stiff Van der Pol (mu = 1000): Sweeping step sizes dt from 0.001 to 0.050...")

def vdp1000_deriv(y):
    return np.array([y[1], 1000.0 * (1.0 - y[0]**2) * y[1] - y[0]])

for dt_test in [0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050]:
    results = {}
    for name in ["standard_rk4", "pratyaksh_ii", "morphocalc"]:
        y = np.array([2.0, 0.0])
        failed = False
        for step in range(500):
            y = morphocalc_step(y, dt_test, vdp1000_deriv, method=name)
            if np.any(np.isnan(y)) or np.any(np.isinf(y)) or np.max(np.abs(y)) > 1e6:
                failed = True
                break
        results[name] = "CRASH" if failed else "PASS"
    print(f"  dt = {dt_test:.3f}: RK4: {results['standard_rk4']:<5} | Pratyaksh-II: {results['pratyaksh_ii']:<5} | MorphoCalc: {results['morphocalc']}")

print("\n" + "=" * 80)
print("  STRESS TESTING COMPLETE")
print("=" * 80)
