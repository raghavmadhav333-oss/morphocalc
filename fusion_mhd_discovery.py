import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# =============================================================================
# 1. Tokamak Plasma DNS Simulator (Magnetohydrodynamic Cascade)
# =============================================================================
class TokamakPlasmaGenerator:
    def __init__(self, N=256, device='cuda'):
        self.N = N
        self.device = device
        
    def generate_mhd_field(self):
        print(f"[Plasma Simulator] Generating {self.N}x{self.N} Tokamak plasma field...")
        print("[Plasma Simulator] Enforcing Alfven-wave kinetic-magnetic equipartition...")
        
        k = torch.arange(-self.N//2, self.N//2, device=self.device)
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        
        k_mag = torch.sqrt(kx**2 + ky**2)
        k_mag[self.N//2, self.N//2] = 1.0 
        
        # Plasma turbulence spectrum
        E_k = k_mag ** (-3.0 / 2.0) # Iroshnikov-Kraichnan spectrum for MHD
        E_k[self.N//2, self.N//2] = 0.0 
        
        # Velocity fields
        phase_u = torch.rand(self.N, self.N, device=self.device) * 2 * np.pi
        phase_v = torch.rand(self.N, self.N, device=self.device) * 2 * np.pi
        u_hat = torch.sqrt(E_k) * torch.exp(1j * phase_u)
        v_hat = torch.sqrt(E_k) * torch.exp(1j * phase_v)
        
        u = torch.real(torch.fft.ifft2(torch.fft.ifftshift(u_hat)))
        v = torch.real(torch.fft.ifft2(torch.fft.ifftshift(v_hat)))
        
        # Magnetic fields (B_x, B_y)
        phase_Bx = torch.rand(self.N, self.N, device=self.device) * 2 * np.pi
        phase_By = torch.rand(self.N, self.N, device=self.device) * 2 * np.pi
        Bx_hat = torch.sqrt(E_k) * torch.exp(1j * phase_Bx)
        By_hat = torch.sqrt(E_k) * torch.exp(1j * phase_By)
        
        Bx = torch.real(torch.fft.ifft2(torch.fft.ifftshift(Bx_hat)))
        By = torch.real(torch.fft.ifft2(torch.fft.ifftshift(By_hat)))
        
        # Normalize
        u, v = u / torch.std(u), v / torch.std(v)
        Bx, By = Bx / torch.std(Bx), By / torch.std(By)
        
        return {'u': u.float(), 'v': v.float(), 'Bx': Bx.float(), 'By': By.float()}

    def compute_plasma_invariants(self, field):
        print("[Plasma Simulator] Extracting Strain Rates and Current Density (J = curl B)...")
        u, v, Bx, By = field['u'], field['v'], field['Bx'], field['By']
        
        u_x, u_y = torch.gradient(u, dim=0)[0], torch.gradient(u, dim=1)[0]
        v_x, v_y = torch.gradient(v, dim=0)[0], torch.gradient(v, dim=1)[0]
        Bx_x, Bx_y = torch.gradient(Bx, dim=0)[0], torch.gradient(Bx, dim=1)[0]
        By_x, By_y = torch.gradient(By, dim=0)[0], torch.gradient(By, dim=1)[0]
        
        S_mag = torch.sqrt(2 * u_x**2 + 2 * v_y**2 + (u_y + v_x)**2 + 1e-8)
        J_current = By_x - Bx_y  # Out-of-plane plasma current density
        
        Delta = torch.full_like(S_mag, 2 * np.pi / self.N)
        
        return {'u_x': u_x, 'v_y': v_y, 'Bx_x': Bx_x, 'By_y': By_y, 
                '|S|': S_mag, 'J': J_current, 'Delta': Delta}

# =============================================================================
# 2. Buckingham Pi: Plasma Dynamics
# =============================================================================
class PhysicalTerm:
    def __init__(self, name, units):
        self.name = name
        self.units = units

def get_plasma_invariants():
    S_mag = PhysicalTerm("|S|", (0, -1, 0))
    J = PhysicalTerm("J", (0, -1, -1)) # simplified Lorentz scaling
    u_x = PhysicalTerm("u_x", (0, -1, 0))
    Bx_x = PhysicalTerm("Bx_x", (0, -1, -1))
    Delta = PhysicalTerm("Delta", (1, 0, 0))
    
    # Target: Maxwell Sub-Grid Stress (Magnetic turbulence leakage)
    target_stress = (2, -2, -2) 
    return [S_mag, J, u_x, Bx_x, Delta], target_stress

def generate_constrained_sindy_library_mhd(base_terms, max_poly_degree, target_units):
    import itertools
    valid_library = []
    for degree in range(1, max_poly_degree + 1):
        for combo in itertools.combinations_with_replacement(base_terms, degree):
            total_units = [sum(term.units[i] for term in combo) for i in range(3)]
            if tuple(total_units) == target_units:
                valid_library.append(combo)
    return valid_library

# =============================================================================
# 3. Differentiable Plasma Engine
# =============================================================================
class DifferentiableMHD(nn.Module):
    def __init__(self, valid_library):
        super().__init__()
        self.library = valid_library
        self.sgs_coefficients = nn.Parameter(torch.zeros(len(valid_library)))
        nn.init.normal_(self.sgs_coefficients, mean=0.01, std=0.001)

    def forward(self, u, v, Bx, By, invariants_map):
        # We are predicting the divergence of the Maxwell sub-grid stress tensor
        sgs_stress = torch.zeros_like(u)
        val_map = {'u': u, 'v': v, 'Bx': Bx, 'By': By}
        val_map.update(invariants_map)
        
        for i, combo in enumerate(self.library):
            term_val = torch.ones_like(u)
            for factor in combo:
                term_val = term_val * val_map[factor.name]
            sgs_stress += self.sgs_coefficients[i] * term_val
                
        return sgs_stress

# =============================================================================
# 4. Master Orchestrator
# =============================================================================
def run_fusion_discovery():
    print("=========================================================")
    print("   MORPHOCALC: NUCLEAR FUSION PLASMA DISCOVERY ENGINE    ")
    print("=========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ingestor = TokamakPlasmaGenerator(N=512, device=device) 
    plasma_data = ingestor.generate_mhd_field()
    invariants_map = ingestor.compute_plasma_invariants(plasma_data)
    
    base_terms, target_stress = get_plasma_invariants()
    valid_library = generate_constrained_sindy_library_mhd(base_terms, max_poly_degree=4, target_units=target_stress)
    
    engine = DifferentiableMHD(valid_library)
    if torch.cuda.device_count() > 1:
        engine = nn.DataParallel(engine)
    engine = engine.to(device)
    
    optimizer = optim.Adam(engine.parameters(), lr=0.01)
    loss_fn = nn.MSELoss() # Simplified for direct tensor matching in phase 1
    
    # "True" magnetic subgrid stress to hunt for (simulating real plasma physics leakage)
    true_maxwell_stress = 0.05 * (invariants_map['Delta']**2) * invariants_map['J'] * invariants_map['Bx_x']
    
    print(f"[AI Training] Searching {len(valid_library)} valid thermodynamic combinations...")
    epochs = 10000 
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        pred_stress = engine(u=plasma_data['u'], v=plasma_data['v'], Bx=plasma_data['Bx'], By=plasma_data['By'], invariants_map=invariants_map)
        loss = loss_fn(pred_stress, true_maxwell_stress)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"  -> Epoch {epoch:05d} | Plasma SGS Leakage Loss: {loss.item():.8f}")
            
    print("\n=========================================================")
    print("   DISCOVERY COMPLETE: EXTRACTED PLASMA CLOSURE MODEL    ")
    print("=========================================================")
    sgs_coeffs = engine.module.sgs_coefficients if isinstance(engine, nn.DataParallel) else engine.sgs_coefficients
    learned_coeffs = sgs_coeffs.detach().cpu().numpy()
    
    for i, combo in enumerate(valid_library):
        term_name = " * ".join([term.name for term in combo])
        coeff = learned_coeffs[i]
        if abs(coeff) > 0.005:
            print(f" -> {coeff:+.4f} * ( {term_name} )")

if __name__ == "__main__":
    run_fusion_discovery()
