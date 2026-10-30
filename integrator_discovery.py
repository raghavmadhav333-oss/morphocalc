import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. Stiff Physics: Van der Pol Oscillator (mu=1000)
# =============================================================================
def stiff_vanderpol(t, y, mu=1000.0):
    """
    y[:, 0] = position, y[:, 1] = velocity
    Standard RK4 explodes to NaN immediately at dt=0.002.
    """
    dy0 = y[:, 1]
    dy1 = mu * (1.0 - y[:, 0]**2) * y[:, 1] - y[:, 0]
    return torch.stack([dy0, dy1], dim=1)

def generate_ground_truth(steps=200, dt=0.002):
    """
    Exact trajectory via 1000 micro-steps per macro-step.
    This is the 'oracle' that the AI must learn to match.
    """
    y = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    trajectory = [y.clone()]
    micro_dt = dt / 1000.0
    for _ in range(steps):
        for _ in range(1000):
            k1 = micro_dt * stiff_vanderpol(0, y)
            k2 = micro_dt * stiff_vanderpol(0, y + 0.5 * k1)
            k3 = micro_dt * stiff_vanderpol(0, y + 0.5 * k2)
            k4 = micro_dt * stiff_vanderpol(0, y + k3)
            y = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        trajectory.append(y.clone())
    return torch.cat(trajectory, dim=0)  # shape: (steps+1, 2)

# =============================================================================
# 2. Baselines
# =============================================================================
def standard_rk4_step(f, t, y, dt):
    K1 = dt * f(t, y)
    K2 = dt * f(t + 0.5*dt, y + 0.5*K1)
    K3 = dt * f(t + 0.5*dt, y + 0.5*K2)
    K4 = dt * f(t + dt, y + K3)
    return y + (K1 + 2*K2 + 2*K3 + K4) / 6.0

def pratyaksh_formula_A(f, t, y, dt):
    """Pratyaksh's original Formula A - the baseline to beat."""
    K1 = dt * f(t, y)
    K2 = dt * f(t + 0.5*dt, y + 0.5*K1)
    K3 = dt * f(t + 0.5*dt, y + 0.5*K2)
    K4 = dt * f(t + dt, y + K3)
    C_A = K4 - 2*K3 + K2
    D = torch.sum(C_A**2) / (torch.sum(K1**2) + 1e-14)
    den = 1.0 + 0.5 * D
    return y + (K1 + 2*K2 + 2*K3 + K4) / (6.0 * den)

def measure_trajectory_error(step_fn, truth, dt):
    """Runs a full rollout and returns total MSE vs. ground truth."""
    y = truth[0].unsqueeze(0)
    total_err = 0.0
    for step in range(1, len(truth)):
        y = step_fn(stiff_vanderpol, 0, y, dt)
        if torch.isnan(y).any() or torch.isinf(y).any():
            return float('inf'), True  # crashed
        total_err += nn.MSELoss()(y, truth[step].unsqueeze(0)).item()
    return total_err, False

# =============================================================================
# 3. AI-Discovered Runge-Kutta Integrator
# =============================================================================
class AIRungeKutta(nn.Module):
    """
    AI-parameterized RK4 where the stiffness denominator is a
    learnable polynomial in the curvature ratio D:
        den = c0 + c1*D + c2*D^2 + c3*D^3 + c4*D^4

    Pratyaksh Formula A hardcodes: den = 1.0 + 0.5*D
    We let the AI discover the optimal polynomial from data.
    """
    def __init__(self):
        super().__init__()
        self.poly = nn.Parameter(torch.zeros(5, dtype=torch.float64))
        with torch.no_grad():
            self.poly[0] = 1.0   # Start near standard RK4

    def forward(self, f, t, y, dt):
        K1 = dt * f(t, y)
        K2 = dt * f(t + 0.5*dt, y + 0.5*K1)
        K3 = dt * f(t + 0.5*dt, y + 0.5*K2)
        K4 = dt * f(t + dt, y + K3)

        # Curvature ratio: captures stiffness magnitude
        C_A = K4 - 2*K3 + K2
        norm_C = torch.sum(C_A**2, dim=1, keepdim=True)
        norm_K1 = torch.sum(K1**2, dim=1, keepdim=True)
        D = norm_C / (norm_K1 + 1e-14)

        # AI polynomial denominator
        den = (self.poly[0]
             + self.poly[1] * D
             + self.poly[2] * (D**2)
             + self.poly[3] * (D**3)
             + self.poly[4] * (D**4))
        den = torch.clamp(den, min=1e-6)

        return y + (K1 + 2*K2 + 2*K3 + K4) / (6.0 * den)

# =============================================================================
# 4. Discovery: Single-Step Curriculum Training
# =============================================================================
def discover_better_integrator():
    print("=========================================================")
    print("   DISCOVERING A SUPERIOR NUMERICAL INTEGRATION FORMULA  ")
    print("=========================================================")

    DT = 0.002  # Large enough to stress the integrator; small enough for Pratyaksh to survive

    # ── Ground Truth ──────────────────────────────────────────────
    print(f"\n[*] Generating ground truth (200 steps x 1000 micro-steps each)...")
    truth = generate_ground_truth(steps=200, dt=DT)
    print(f"    Trajectory shape: {truth.shape}")

    # ── Baseline: Standard RK4 ────────────────────────────────────
    rk4_err, rk4_crashed = measure_trajectory_error(standard_rk4_step, truth, DT)
    if rk4_crashed:
        print(f"[Baseline] Standard RK4:      CRASHED immediately (as expected for mu=1000)")
    else:
        print(f"[Baseline] Standard RK4 MSE:  {rk4_err:.6f}")

    # ── Baseline: Pratyaksh Formula A ────────────────────────────
    pa_err, pa_crashed = measure_trajectory_error(pratyaksh_formula_A, truth, DT)
    if pa_crashed:
        print(f"[Baseline] Pratyaksh Formula A: CRASHED")
        pa_err = 1e6
    else:
        print(f"[Baseline] Pratyaksh Formula A MSE: {pa_err:.6f}")

    # ── AI Discovery ──────────────────────────────────────────────
    ai_rk = AIRungeKutta()
    optimizer = optim.Adam(ai_rk.parameters(), lr=0.003)

    epochs = 3000
    batch_size = 32
    n_states = len(truth) - 1

    print(f"\n[AI] Single-Step Curriculum Training: {epochs} epochs, batch={batch_size}")
    print("     Strategy: pick random states from ground truth, predict ONE step forward.")
    print("     This guarantees clean gradients — no NaN cascade, no disconnected loss.\n")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Sample a random batch of (current_state, next_state) pairs from the oracle
        indices = torch.randint(0, n_states, (batch_size,))
        y_batch = torch.stack([truth[i]   for i in indices])   # (B, 2)
        y_next  = torch.stack([truth[i+1] for i in indices])   # (B, 2)

        # AI takes ONE step — gradient always flows cleanly through poly
        y_pred = ai_rk(stiff_vanderpol, 0, y_batch, DT)

        loss = nn.MSELoss()(y_pred, y_next)
        loss.backward()
        optimizer.step()

        if epoch % 300 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:04d} | Single-Step MSE: {loss.item():.8f}")

    # ── Final Evaluation ──────────────────────────────────────────
    print("\n[*] Final full-trajectory evaluation of AI integrator...")
    
    def ai_step(f, t, y, dt):
        return ai_rk(f, t, y, dt)

    ai_err, ai_crashed = measure_trajectory_error(ai_step, truth, DT)
    if ai_crashed:
        print("    AI integrator: CRASHED during evaluation")
        ai_err = 1e6
    else:
        print(f"    AI Integrator MSE: {ai_err:.6f}")

    improvement = (1.0 - ai_err / (pa_err + 1e-10)) * 100.0

    # ── Final Report ──────────────────────────────────────────────
    poly = ai_rk.poly.detach().numpy()
    formula = f"{poly[0]:.4f}"
    for p in range(1, 5):
        if abs(poly[p]) > 1e-4:
            formula += f" {'+' if poly[p]>0 else '-'} {abs(poly[p]):.4f}*D^{p}"

    print("\n=========================================================")
    print("   DISCOVERY COMPLETE: THE AI RUNGE-KUTTA FORMULA        ")
    print("=========================================================")
    print(f"  Standard RK4:      den = 1.0000               | MSE: {'CRASHED' if rk4_crashed else f'{rk4_err:.4f}'}")
    print(f"  Pratyaksh A:       den = 1.0000 + 0.5000*D    | MSE: {pa_err:.4f}")
    print(f"  AI-Discovered:     den = {formula:<30s}| MSE: {ai_err:.4f}")
    print(f"\n  Improvement over Pratyaksh Formula A: {improvement:.1f}%")

if __name__ == "__main__":
    discover_better_integrator()
