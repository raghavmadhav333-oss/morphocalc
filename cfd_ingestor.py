import torch
import numpy as np
import h5py
import pandas as pd
import os

class CFDIngestor:
    """
    A robust data pipeline for loading real-world Computational Fluid Dynamics (CFD) 
    datasets into the MorphoCalc PyTorch engine.
    """
    def __init__(self, device='cpu'):
        self.device = device

    def load_hdf5_grid(self, filepath, velocity_keys=['u', 'v'], time_key='t', spatial_keys=['x', 'y']):
        """
        Loads structured 2D CFD data from an HDF5 file (standard OpenFOAM export format).
        Returns PyTorch tensors ready for differentiable physics.
        """
        print(f"[CFD Ingestor] Loading HDF5 dataset from {filepath}...")
        
        with h5py.File(filepath, 'r') as f:
            # Load basic grid
            x = torch.tensor(f[spatial_keys[0]][:], dtype=torch.float32, device=self.device)
            y = torch.tensor(f[spatial_keys[1]][:], dtype=torch.float32, device=self.device)
            t = torch.tensor(f[time_key][:], dtype=torch.float32, device=self.device)
            
            # Load velocity fields (Time, X, Y)
            u = torch.tensor(f[velocity_keys[0]][:], dtype=torch.float32, device=self.device)
            v = torch.tensor(f[velocity_keys[1]][:], dtype=torch.float32, device=self.device)
            
        print(f"[CFD Ingestor] Loaded {len(t)} timesteps across a {len(x)}x{len(y)} spatial grid.")
        return u, v, x, y, t

    def compute_spatial_derivatives_2d(self, u, v, dx, dy):
        """
        Pre-computes all necessary spatial derivatives (u_x, u_y, v_x, v_y, u_xx, v_yy)
        using PyTorch finite differences across the 2D grid.
        
        u and v should be of shape (Time, X, Y)
        """
        print("[CFD Ingestor] Pre-computing 2D spatial derivatives via central differences...")
        
        # u_x: derivative of u with respect to x (dim 1)
        u_x = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
        # u_y: derivative of u with respect to y (dim 2)
        u_y = (torch.roll(u, -1, dims=2) - torch.roll(u, 1, dims=2)) / (2 * dy)
        
        # v_x, v_y
        v_x = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dx)
        v_y = (torch.roll(v, -1, dims=2) - torch.roll(v, 1, dims=2)) / (2 * dy)
        
        # Laplacians (2nd derivatives)
        u_xx = (torch.roll(u, -1, dims=1) - 2*u + torch.roll(u, 1, dims=1)) / (dx**2)
        u_yy = (torch.roll(u, -1, dims=2) - 2*u + torch.roll(u, 1, dims=2)) / (dy**2)
        v_xx = (torch.roll(v, -1, dims=1) - 2*v + torch.roll(v, 1, dims=1)) / (dx**2)
        v_yy = (torch.roll(v, -1, dims=2) - 2*v + torch.roll(v, 1, dims=2)) / (dy**2)
        
        print("[CFD Ingestor] Derivatives computed successfully.")
        
        derivatives = {
            'u_x': u_x, 'u_y': u_y,
            'v_x': v_x, 'v_y': v_y,
            'u_xx': u_xx, 'u_yy': u_yy,
            'v_xx': v_xx, 'v_yy': v_yy
        }
        return derivatives

# Example Usage
if __name__ == "__main__":
    print("CFDIngestor is ready to process real-world 2D datasets.")
