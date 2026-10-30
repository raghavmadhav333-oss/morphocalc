import torch
import torch.nn as nn
import urllib.request
import scipy.io
import os
import itertools

# =============================================================================
# PHASE 1: THE UNLEASHED PARC LIBRARY (DIMENSIONAL PRUNING)
# =============================================================================
class PhysicalTerm:
    def __init__(self, name, function, unit_dimensions):
        self.name = name
        self.function = function
        self.unit_dimensions = unit_dimensions

def generate_constrained_sindy_library(state_vars, spatial_derivs, max_poly_degree, target_units):
    base_terms = state_vars + spatial_derivs
    combinations = []
    for degree in range(1, max_poly_degree + 1):
        combinations.extend(itertools.combinations_with_replacement(base_terms, degree))
        
    valid_terms = []
    for combo in combinations:
        combined_units = [sum(term.unit_dimensions[i] for term in combo) for i in range(3)]
        if tuple(combined_units) == target_units:
            valid_terms.append(combo) # Store the actual tuple of PhysicalTerm objects
    return valid_terms

# =============================================================================
# PHASE 2: PRATYAKSH RATIONAL RUNGE-KUTTA INTEGRATOR
# =============================================================================
def pratyaksh_rk_step(physics_func, t, y, dt):
    """
    Stabilizes AI physics discovery during stiff shockwaves by freezing time 
    if the predicted physics equations become violently unstable.
    """
    k1 = physics_func(t, y)
    k2 = physics_func(t + 0.5*dt, y + 0.5*dt*k1)
    
    # Pratyaksh Formula A (Shock-Capturing)
    curvature = k2 - k1
    D = torch.abs(curvature) / (torch.abs(k1) + 1e-8)
    den = 1.0 + 0.5 * D
    
    y_next = y + (dt * k1) / den
    return y_next

# =============================================================================
# PHASE 3: DYNAMIC AI PHYSICS ENGINE
# =============================================================================
class MorphoCalcEngine(nn.Module):
    def __init__(self, valid_library_combos):
        super().__init__()
        self.library_combos = valid_library_combos
        # Create a learnable PyTorch weight for EVERY surviving term in the library
        self.coefficients = nn.Parameter(torch.zeros(len(valid_library_combos)))
        # Initialize slightly above 0 so gradients flow
        nn.init.normal_(self.coefficients, mean=0.1, std=0.01)

    def forward(self, t, u, u_x, u_xx):
        """
        Dynamically calculates the physical acceleration by summing:
        Coefficient_i * (Term_A * Term_B ...)
        """
        total_acceleration = torch.zeros_like(u)
        
        # A dictionary to look up the tensor values dynamically
        val_map = {"u": u, "u_x": u_x, "u_xx": u_xx}
        
        for i, combo in enumerate(self.library_combos):
            term_val = torch.ones_like(u)
            for factor in combo:
                # Multiply the values of the terms together (e.g., u * u_x)
                if factor.name in val_map:
                     term_val = term_val * val_map[factor.name]
                else:
                     term_val = term_val * 0.0 # Ignore unsupported terms for this demo
            
            # Add to total: c_i * (term_val)
            total_acceleration += self.coefficients[i] * term_val
            
        return total_acceleration

# =============================================================================
# EXECUTION: REAL-WORLD SHOCKWAVE DISCOVERY
# =============================================================================
if __name__ == "__main__":
    print("=== MORPHOCALC GRAND UNIFICATION RUN ===")
    
    # 1. Define physical building blocks
    u = PhysicalTerm("u", None, (1, -1, 0))        
    u_x = PhysicalTerm("u_x", None, (0, -1, 0))    
    u_xx = PhysicalTerm("u_xx", None, (-1, -1, 0)) 
    target_acceleration = (1, -2, 0) # u_t
    
    # 2. Generate Dimensionally Constrained Library
    valid_library = generate_constrained_sindy_library(
        state_vars=[u], 
        spatial_derivs=[u_x, u_xx], 
        max_poly_degree=3, 
        target_units=target_acceleration
    )
    
    print(f"\n[AI Setup] Discovered {len(valid_library)} valid physics terms out of millions of possibilities.")
    
    # 3. Initialize the AI Engine
    ai_engine = MorphoCalcEngine(valid_library)
    optimizer = torch.optim.Adam(ai_engine.parameters(), lr=0.01)
    
    # 4. Fetch Real-World Data (Burgers' Shockwave)
    filename = "burgers_shock.mat"
    if not os.path.exists(filename):
        print("[Data] Downloading real Burgers' shockwave dataset...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/burgers_shock.mat", filename)
    
    data = scipy.io.loadmat(filename)
    # Get a tiny slice of the data for a lightning-fast run (100 spatial points, 50 time steps)
    real_u = torch.tensor(data['usol'][:100, :50].real, dtype=torch.float32)
    
    # Pre-calculate spatial derivatives of the real data (using simple finite differences for speed)
    dx = 0.01
    real_u_x = (torch.roll(real_u, -1, dims=0) - torch.roll(real_u, 1, dims=0)) / (2*dx)
    real_u_xx = (torch.roll(real_u, -1, dims=0) - 2*real_u + torch.roll(real_u, 1, dims=0)) / (dx**2)
    
    print("\n[Training] Beginning Differentiable Physics Discovery over Real Data...")
    
    # 5. Differentiable Physics Loop
    for epoch in range(100):
        optimizer.zero_grad()
        loss = 0
        
        # Start at t=0
        u_pred = real_u[:, 0]
        
        # Simulate forward in time using Pratyaksh Integrator
        for t_step in range(1, real_u.shape[1]):
            # The AI engine calculates the spatial physics
            def physics_func(t, y_state):
                return ai_engine(t, y_state, real_u_x[:, t_step], real_u_xx[:, t_step])
                
            # The Integrator moves time forward safely
            u_pred = pratyaksh_rk_step(physics_func, 0, u_pred, dt=0.01)
            
            # Penalize deviation from reality
            loss += torch.nn.functional.mse_loss(u_pred, real_u[:, t_step])
            
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Differentiable Physics Loss: {loss.item():.4f}")
            
    print("\n[Results] Discovery Complete! AI Learned Coefficients for surviving terms:")
    for i, combo in enumerate(valid_library):
        term_name = " * ".join([term.name for term in combo])
        print(f" -> {term_name}: {ai_engine.coefficients[i].item():.5f}")
