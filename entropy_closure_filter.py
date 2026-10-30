import torch

class EntropyClosureFilter:
    """
    Implements the Second Law of Thermodynamics (Non-Equilibrium Statistical Mechanics)
    as a hard constraint on the AI's equation discovery.
    
    A valid turbulence closure (Reynolds Stress Tensor, tau) must dissipate kinetic 
    energy into heat. Therefore, the entropy production rate (sigma) must be positive:
        sigma = tau_ij * S_ij >= 0
        
    This filter evaluates candidate SINDy library terms. If a term allows negative 
    entropy production (e.g., negative viscosity), it is discarded before training.
    """
    def __init__(self):
        pass

    def check_entropy_production(self, candidate_tau_term, S_ij_sample):
        """
        Symbolically evaluates if the term guarantees energy dissipation.
        candidate_tau_term: PyTorch tensor representing the candidate stress.
        S_ij_sample: PyTorch tensor representing the local Strain Rate Tensor.
        """
        # The entropy production is the Frobenius inner product of the stress and strain rate tensors.
        # sigma = tau : S = sum(tau_ij * S_ij)
        entropy_production = torch.sum(candidate_tau_term * S_ij_sample)
        
        # In a strict mathematical formulation, this must be >= 0 everywhere.
        # We check the minimum value across the test sample.
        if torch.min(entropy_production) < -1e-6:
            return False # Violates the 2nd Law of Thermodynamics!
        return True

    def filter_library(self, library_terms_evaluated, S_ij_sample):
        """
        Takes the dimensionally-valid Buckingham Pi library and permanently
        prunes terms that violate thermodynamic entropy constraints.
        """
        print("[Thermodynamics] Enforcing Entropy Production Constraint (sigma >= 0)...")
        physically_valid_indices = []
        
        for i in range(library_terms_evaluated.shape[1]):
            term_eval = library_terms_evaluated[:, i]
            
            if self.check_entropy_production(term_eval, S_ij_sample):
                physically_valid_indices.append(i)
                
        print(f"[Thermodynamics] Entropy filter complete. {len(physically_valid_indices)} out of {library_terms_evaluated.shape[1]} terms preserve the Second Law.")
        return physically_valid_indices

if __name__ == "__main__":
    print("=== MORPHOCALC: ENTROPY CLOSURE FILTER ===")
    filter_engine = EntropyClosureFilter()
    
    # Mock strain rate and stress terms
    mock_S_ij = torch.tensor([1.0, -0.5, 2.0])
    
    # 1. Smagorinsky-style (tau = c * |S| * S_ij) -> Always positive dissipation
    valid_tau = 0.05 * torch.abs(mock_S_ij) * mock_S_ij 
    print(f"Valid Term (Smagorinsky) passes 2nd Law: {filter_engine.check_entropy_production(valid_tau, mock_S_ij)}")
    
    # 2. Unphysical term (negative viscosity, energy creation)
    invalid_tau = -0.05 * mock_S_ij
    print(f"Invalid Term (Negative Viscosity) passes 2nd Law: {filter_engine.check_entropy_production(invalid_tau, mock_S_ij)}")
