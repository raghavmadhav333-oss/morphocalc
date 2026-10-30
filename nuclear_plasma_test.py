import itertools

class PhysicalTerm:
    def __init__(self, name, unit_dimensions):
        self.name = name
        self.unit_dimensions = unit_dimensions

def generate_constrained_sindy_library(base_terms, max_poly_degree, target_units):
    combinations = []
    for degree in range(1, max_poly_degree + 1):
        combinations.extend(itertools.combinations_with_replacement(base_terms, degree))
        
    valid_terms = []
    rejected_count = 0
    for combo in combinations:
        # Sum units for length, time, mass, electric current (L, T, M, I)
        combined_units = [sum(term.unit_dimensions[i] for term in combo) for i in range(4)]
        
        if tuple(combined_units) == target_units:
            term_name = " * ".join([term.name for term in combo])
            valid_terms.append(term_name)
        else:
            rejected_count += 1
            
    print(f"--- NUCLEAR FUSION (PLASMA) PRUNING RESULTS ---")
    print(f"Total possible plasma interactions generated: {len(combinations)}")
    print(f"Physically IMPOSSIBLE plasma interactions rejected: {rejected_count}")
    print(f"Physically VALID interactions kept: {len(valid_terms)}\n")
    return valid_terms

if __name__ == "__main__":
    print("=== MORPHOCALC: NUCLEAR FUSION (MAGNETOHYDRODYNAMICS) TEST ===")
    
    # In a Tokamak Nuclear Reactor, the physics uses 4 base dimensions: (Length, Time, Mass, Current)
    
    # Let's define the building blocks of a 100-million degree plasma:
    B = PhysicalTerm("B", (0, -2, 1, -1))         # Magnetic Field (Tesla)
    B_x = PhysicalTerm("B_x", (-1, -2, 1, -1))    # Magnetic Gradient
    B_xx = PhysicalTerm("B_xx", (-2, -2, 1, -1))  # Magnetic Laplacian (Diffusion)
    
    u = PhysicalTerm("u", (1, -1, 0, 0))          # Plasma Velocity (m/s)
    u_x = PhysicalTerm("u_x", (0, -1, 0, 0))      # Plasma Velocity Gradient
    
    rho = PhysicalTerm("rho", (-3, 0, 1, 0))      # Plasma Density (kg/m^3)
    J = PhysicalTerm("J", (-2, 0, 0, 1))          # Electric Current Density (A/m^2)
    
    base_terms = [B, B_x, B_xx, u, u_x, rho, J]
    
    # Goal: Discover the Magnetic Induction Equation!
    # We want to know how the Magnetic Field changes over time (B_t)
    # Units of B_t = Tesla / second = (0, -3, 1, -1)
    target_magnetic_evolution = (0, -3, 1, -1)
    
    print("\nTarget: Discovering equations for Magnetic Field Evolution (B_t)...")
    valid_library = generate_constrained_sindy_library(
        base_terms, 
        max_poly_degree=3, 
        target_units=target_magnetic_evolution
    )
    
    print("\nValid terms surviving the Buckingham Pi dimensional filter:")
    for term in valid_library:
        print(f" -> {term}")
    
    print("\n(Note: The filter successfully isolates the 1D product-rule expansion of the")
    print("advective flux term d(uB)/dx. This demonstrates effective algorithmic bookkeeping")
    print("and dimensional homogeneity filtering to shrink the SINDy candidate library.)")
