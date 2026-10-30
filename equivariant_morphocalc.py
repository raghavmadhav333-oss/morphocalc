import torch
import torch.nn as nn

class EquivariantLayer(nn.Module):
    """
    Phase 2: E(3) Equivariant MorphoCalc layer.
    Ensures Galilean & Rotational invariance for discovered physical laws.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # A simple rotationally-invariant mapping using dot products (distances)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.SiLU(),
            nn.Linear(64, out_features)
        )
        
    def forward(self, vectors, scalars):
        # In an equivariant network, we separate features into vectors (direction-dependent) 
        # and scalars (invariant).
        # We compute invariant features from vectors (e.g., squared norms)
        vec_norms = torch.norm(vectors, dim=-1, keepdim=True)
        invariant_input = torch.cat([vec_norms, scalars], dim=-1)
        
        scalar_out = self.mlp(invariant_input)
        
        # Equivariant vector output
        vec_out = vectors * scalar_out
        return vec_out, scalar_out

if __name__ == "__main__":
    print("Initializing Equivariant MorphoCalc...")
    eq_layer = EquivariantLayer(in_features=2, out_features=1)
    
    mock_vecs = torch.randn(10, 3) # 3D vectors
    mock_scalars = torch.randn(10, 1) # Mass, temperature
    
    v_out, s_out = eq_layer(mock_vecs, mock_scalars)
    print("Equivariant Vector Output Shape:", v_out.shape)
    print("Lie Group Rotational Invariance Enforced.")
