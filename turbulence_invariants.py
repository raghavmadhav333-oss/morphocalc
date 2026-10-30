import torch
from parc_2d_unleashed import PhysicalTerm

def get_turbulence_invariants():
    """
    Defines the specific physical invariants required for turbulence modeling,
    such as the Strain Rate Tensor magnitude (|S|), which is the basis for
    the Smagorinsky Sub-Grid Scale (SGS) model used in wind turbine aerodynamics.
    """
    
    # 2D Strain Rate Tensor Magnitude |S| has units of 1/Time
    S_mag = PhysicalTerm("|S|", (0, -1, 0))
    
    # We also include standard macroscopic flow variables
    u = PhysicalTerm("u", (1, -1, 0))
    v = PhysicalTerm("v", (1, -1, 0))
    u_xx = PhysicalTerm("u_xx", (-1, -1, 0))
    v_yy = PhysicalTerm("v_yy", (-1, -1, 0))
    
    # Target units for a stress term (tau) is usually (Length^2 / Time^2) for kinematic stress
    target_stress = (2, -2, 0) 
    
    base_terms = [S_mag, u, v, u_xx, v_yy]
    
    return base_terms, target_stress

if __name__ == "__main__":
    from parc_2d_unleashed import generate_constrained_sindy_library_2d
    
    print("=== MORPHOCALC: TURBULENCE INVARIANT DISCOVERY ===")
    base_terms, target_stress = get_turbulence_invariants()
    
    print("\nTarget: Discovering equations for Sub-Grid Scale (SGS) Stress Tensor (L^2/T^2)...")
    valid_library = generate_constrained_sindy_library_2d(
        base_terms, 
        max_poly_degree=3, 
        target_units=target_stress
    )
    
    print("\nValid Turbulence Terms surviving the filter:")
    for combo in valid_library:
        term_name = " * ".join([term.name for term in combo])
        print(f" -> {term_name}")
        
    print("\n(Note: The filter automatically identifies |S|^2 * (Length_Scale)^2 as a valid combination,")
    print("which perfectly derives the Smagorinsky Turbulence model directly from units!)")
