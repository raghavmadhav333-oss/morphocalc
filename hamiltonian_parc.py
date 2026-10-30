import torch
import torch.nn as nn
from torch.autograd import grad

class HamiltonianNeuralNetwork(nn.Module):
    """
    Phase 2: Universal Core (Mathematical Symmetry)
    Enforces Noether's Theorem by learning the Hamiltonian (Energy Action) 
    instead of raw state mappings.
    """
    def __init__(self):
        super().__init__()
        # Input is phase space [q, p] (position, momentum)
        self.H_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Outputs a scalar Energy (Hamiltonian)
        )
        
    def forward(self, x):
        # x is [q, p]
        x.requires_grad_(True)
        H = self.H_net(x)
        
        # Compute symplectic gradients: dq/dt = dH/dp, dp/dt = -dH/dq
        dH = grad(H.sum(), x, create_graph=True)[0]
        
        dH_dq = dH[:, 0:1]
        dH_dp = dH[:, 1:2]
        
        # Return time derivatives [dq/dt, dp/dt]
        return torch.cat([dH_dp, -dH_dq], dim=1)

if __name__ == "__main__":
    print("Initializing Hamiltonian PARC...")
    hnn = HamiltonianNeuralNetwork()
    # Mock Phase Space [q, p]
    mock_qp = torch.randn(10, 2) 
    q_dot_p_dot = hnn(mock_qp)
    print("Symplectic Time Derivatives Shape:", q_dot_p_dot.shape)
    print("Noether's Theorem Enforced: Energy is conserved by design.")
