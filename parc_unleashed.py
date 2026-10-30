import torch
import itertools

# =============================================================================
# THE UNLEASHED PARC LIBRARY GENERATOR (With DeepMind Dimensional Constraints)
# =============================================================================

class PhysicalTerm:
    def __init__(self, name, function, unit_dimensions):
        """
        unit_dimensions is a tuple: (Length, Time, Mass)
        e.g., Velocity (m/s) -> (1, -1, 0)
        e.g., Acceleration (m/s^2) -> (1, -2, 0)
        """
        self.name = name
        self.function = function
        self.unit_dimensions = unit_dimensions

def generate_constrained_sindy_library(state_vars, spatial_derivs, max_poly_degree, target_units):
    """
    Generates a massive SINDy library (10,000+ potential terms) but INSTANTLY 
    prunes them based on DeepMind-inspired Dimensional Constraints (Phi-SO).
    """
    print(f"Target Units required: L^{target_units[0]} T^{target_units[1]} M^{target_units[2]}")
    
    base_terms = state_vars + spatial_derivs
    library = []
    
    print(f"Generating polynomial combinations up to degree {max_poly_degree}...")
    
    # Generate all combinations of terms up to max_poly_degree
    combinations = []
    for degree in range(1, max_poly_degree + 1):
        combinations.extend(itertools.combinations_with_replacement(base_terms, degree))
        
    valid_terms = []
    rejected_count = 0
    
    for combo in combinations:
        # Calculate combined units by adding exponents: (L1+L2, T1+T2, M1+M2)
        combined_units = [sum(term.unit_dimensions[i] for term in combo) for i in range(3)]
        
        # Build the mathematical name (e.g., "u * u_x")
        term_name = " * ".join([term.name for term in combo])
        
        # DEEPMIND DIMENSIONAL FILTER: If it doesn't match reality, KILL IT immediately.
        if tuple(combined_units) == target_units:
            valid_terms.append(term_name)
        else:
            rejected_count += 1
            
    print(f"--- DIMENSIONAL PRUNING RESULTS ---")
    print(f"Total possible terms generated: {len(combinations)}")
    print(f"Physically IMPOSSIBLE terms rejected: {rejected_count}")
    print(f"Physically VALID terms kept: {len(valid_terms)}")
    
    return valid_terms

# =============================================================================
# TEST THE FRAMEWORK
# =============================================================================
if __name__ == "__main__":
    # Let's define the fundamental building blocks of a fluid (Navier-Stokes)
    # Units: (Length, Time, Mass)
    
    u = PhysicalTerm("u", None, (1, -1, 0))          # Velocity (m/s)
    u_x = PhysicalTerm("u_x", None, (0, -1, 0))      # Gradient (1/s)
    u_xx = PhysicalTerm("u_xx", None, (-1, -1, 0))   # Laplacian (1/(m*s))
    p_x = PhysicalTerm("p_x", None, (-1, -2, 1))     # Pressure gradient (kg/(m^2*s^2))
    rho_inv = PhysicalTerm("1/rho", None, (3, 0, -1))# Inverse Density (m^3/kg)
    
    # We want to discover the equation for Acceleration (u_t) which is m/s^2 -> (1, -2, 0)
    target_acceleration = (1, -2, 0)
    
    # Generate a library mixing these up to Polynomial Degree 3
    valid_library = generate_constrained_sindy_library(
        state_vars=[u, rho_inv], 
        spatial_derivs=[u_x, u_xx, p_x], 
        max_poly_degree=3, 
        target_units=target_acceleration
    )
    
    print("\nValid Terms surviving the filter (The surviving SINDy Library):")
    for term in valid_library:
        print(f" -> {term}")
