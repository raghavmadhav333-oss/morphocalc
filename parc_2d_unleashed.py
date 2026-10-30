import itertools

class PhysicalTerm:
    def __init__(self, name, unit_dimensions):
        """
        unit_dimensions is a tuple: (Length, Time, Mass)
        """
        self.name = name
        self.unit_dimensions = unit_dimensions

def generate_constrained_sindy_library_2d(base_terms, max_poly_degree, target_units):
    """
    Generates a 2D SINDy library for complex CFD problems (Navier-Stokes) and prunes them.
    """
    combinations = []
    for degree in range(1, max_poly_degree + 1):
        combinations.extend(itertools.combinations_with_replacement(base_terms, degree))
        
    valid_terms = []
    rejected_count = 0
    
    for combo in combinations:
        # Calculate combined units: (L, T, M)
        combined_units = [sum(term.unit_dimensions[i] for term in combo) for i in range(3)]
        
        if tuple(combined_units) == target_units:
            valid_terms.append(combo)
        else:
            rejected_count += 1
            
    print(f"--- 2D NAVIER-STOKES DIMENSIONAL PRUNING ---")
    print(f"Total combinations evaluated: {len(combinations)}")
    print(f"Physically impossible combinations rejected: {rejected_count}")
    print(f"Valid terms retained for CFD engine: {len(valid_terms)}")
    
    return valid_terms

if __name__ == "__main__":
    # 2D Navier-Stokes variables
    u = PhysicalTerm("u", (1, -1, 0))          # X-Velocity
    v = PhysicalTerm("v", (1, -1, 0))          # Y-Velocity
    
    u_x = PhysicalTerm("u_x", (0, -1, 0))      # X-grad of u
    u_y = PhysicalTerm("u_y", (0, -1, 0))      # Y-grad of u
    v_x = PhysicalTerm("v_x", (0, -1, 0))      # X-grad of v
    v_y = PhysicalTerm("v_y", (0, -1, 0))      # Y-grad of v
    
    u_xx = PhysicalTerm("u_xx", (-1, -1, 0))   # X-Laplacian of u
    u_yy = PhysicalTerm("u_yy", (-1, -1, 0))   # Y-Laplacian of u
    v_xx = PhysicalTerm("v_xx", (-1, -1, 0))   # X-Laplacian of v
    v_yy = PhysicalTerm("v_yy", (-1, -1, 0))   # Y-Laplacian of v
    
    p_x = PhysicalTerm("p_x", (-1, -2, 1))     # Pressure gradient X
    p_y = PhysicalTerm("p_y", (-1, -2, 1))     # Pressure gradient Y
    rho_inv = PhysicalTerm("1/rho", (3, 0, -1))# Inverse Density
    
    base_terms = [u, v, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y, rho_inv]
    
    target_acceleration = (1, -2, 0) # e.g., u_t or v_t
    
    valid_library = generate_constrained_sindy_library_2d(
        base_terms, 
        max_poly_degree=3, 
        target_units=target_acceleration
    )
    
    print("\nSurviving 2D Physics Terms:")
    for combo in valid_library:
        term_name = " * ".join([term.name for term in combo])
        print(f" -> {term_name}")
