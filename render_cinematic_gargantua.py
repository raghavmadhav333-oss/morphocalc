"""
Cinematic Gargantua Ray-Tracer (Matching Interstellar 2014 Movie Frame)
Lead Researcher: Raghav
Engine: MorphoCalc Mark-IV Anisotropic Tensor-Padé Integrator
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Device: {device.type.upper()}")

def gr_acceleration(pos, vel, M=1.0):
    r = torch.norm(pos, dim=-1, keepdim=True) + 1e-9
    r_safe = torch.clamp(r, min=2.01 * M)
    a_newton = -M / (r_safe**2)
    L_sq = torch.norm(torch.cross(pos, vel, dim=-1), dim=-1, keepdim=True)**2
    a_gr = -3.0 * M * L_sq / (r_safe**4)
    return (a_newton + a_gr) * (pos / r)

def mark4_step(state, dt, accel_fn, M=1.0):
    pos, vel = state
    k1_v = vel
    k1_a = accel_fn(pos, vel, M)
    k2_v = vel + 0.5 * dt * k1_a
    k2_a = accel_fn(pos + 0.5 * dt * k1_v, k2_v, M)
    k3_v = vel + 0.5 * dt * k2_a
    k3_a = accel_fn(pos + 0.5 * dt * k2_v, k3_v, M)
    k4_v = vel + dt * k3_a
    k4_a = accel_fn(pos + dt * k3_v, k4_v, M)

    C_v = k4_v - k3_v - k2_v + k1_v
    C_a = k4_a - k3_a - k2_a + k1_a
    D_v = (C_v**2) / (k1_v**2 + 1e-12)
    D_a = (C_a**2) / (k1_a**2 + 1e-12)

    den_v = 1.0 + (0.04 * D_v**2) / (1.0 + 0.01 * D_v + 0.0001 * D_v**2)
    den_a = 1.0 + (0.04 * D_a**2) / (1.0 + 0.01 * D_a + 0.0001 * D_a**2)

    new_pos = pos + (dt / 6.0) * ((k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / den_v)
    new_vel = vel + (dt / 6.0) * ((k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a) / den_a)

    v_norm = torch.norm(new_vel, dim=-1, keepdim=True) + 1e-12
    new_vel = new_vel / v_norm
    return new_pos, new_vel

def render_interstellar_gargantua(res_y=400, res_x=720, max_steps=240, dt=0.07, output_path="d:/Research/morphocalc/fig_cinematic_gargantua_interstellar_match.png"):
    print(f"[*] Rendering Cinematic Gargantua ({res_x}x{res_y} = {res_x*res_y:,} rays) with Mark-IV...")
    t0 = time.time()
    
    # Camera matching the Interstellar frame: inclined at 7.5 degrees, slight roll/yaw
    cam_dist = 22.0
    fov = 0.52
    aspect = res_x / res_y
    
    y_screen, x_screen = torch.meshgrid(
        torch.linspace(-fov, fov, res_y, device=device),
        torch.linspace(-fov * aspect, fov * aspect, res_x, device=device),
        indexing='ij'
    )
    
    # 7.5 deg inclination
    theta_cam = np.radians(7.5)
    cam_pos = torch.tensor([cam_dist * np.cos(theta_cam), 0.0, cam_dist * np.sin(theta_cam)], device=device, dtype=torch.float32)
    
    # Ray directions
    ray_dir = torch.stack([
        -torch.ones_like(x_screen),
        x_screen,
        y_screen
    ], dim=-1)
    ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
    
    N_rays = res_y * res_x
    p = cam_pos.unsqueeze(0).expand(N_rays, 3).clone()
    v = ray_dir.reshape(N_rays, 3).clone()
    
    intensity = torch.zeros(N_rays, device=device)
    active = torch.ones(N_rays, dtype=torch.bool, device=device)
    prev_z = p[:, 2].clone()
    
    M = 1.0
    r_inner = 2.6 * M
    r_isco = 6.0 * M
    r_out = 16.5 * M
    
    for step in range(max_steps):
        if not active.any(): break
        curr_idx = torch.nonzero(active, as_tuple=True)[0]
        p_act = p[curr_idx]
        v_act = v[curr_idx]
        
        new_p, new_v = mark4_step((p_act, v_act), dt, gr_acceleration, M)
        p[curr_idx] = new_p
        v[curr_idx] = new_v
        
        r = torch.norm(new_p, dim=-1)
        curr_z = new_p[:, 2]
        crossed = (prev_z[curr_idx] * curr_z < 0.0)
        crossed_idx = curr_idx[crossed]
        r_cross = r[crossed]
        
        # Disk intersection
        in_disk = (r_cross >= r_inner) & (r_cross <= r_out)
        if in_disk.any():
            disk_idx = crossed_idx[in_disk]
            r_hit = r_cross[in_disk]
            phi_hit = torch.atan2(p[disk_idx, 1], p[disk_idx, 0])
            
            # Base Shakura-Sunyaev thermal falloff
            base_lum = (r_hit / r_isco)**(-2.2)
            
            # Fine Keplerian shear striations (filaments)
            filaments = 1.0 + 0.32 * torch.sin(28.0 * torch.log(r_hit + 0.1) + 4.0 * phi_hit)
            
            # Razor-sharp white-hot inner photon ring (n=1)
            photon_ring = 3.2 * torch.exp(-((r_hit - 3.0 * M) / 0.12)**2)
            
            # Hollywood / Nolan balanced Doppler factor (mild asymmetry so both arches glow brightly)
            v_orb = torch.sqrt(torch.tensor(M, device=device) / r_hit)
            sin_phi = -p[disk_idx, 1] / r_hit
            doppler = (1.0 + 0.65 * v_orb * sin_phi)**2.2
            
            intensity[disk_idx] += (base_lum * filaments + photon_ring) * doppler
            
        captured = r < 2.05 * M
        escaped = r > 35.0 * M
        active[curr_idx[captured | escaped]] = False
        prev_z[curr_idx] = curr_z
        
    elapsed = time.time() - t0
    print(f"[+] Render completed in {elapsed:.2f} seconds ({N_rays / elapsed:,.0f} rays/sec)!")
    
    # Tone mapping & cinematic coloring matching Interstellar frame
    img = intensity.reshape(res_y, res_x).cpu().numpy()
    p99 = np.percentile(img, 99.7) + 1e-9
    norm_img = np.clip(img / p99, 0.0, 1.0)
    
    # Custom 3-channel Interstellar color palette:
    # 0.0 -> pitch black
    # 0.2 -> deep amber/bronze
    # 0.6 -> warm gold/orange
    # 0.95 -> brilliant incandescent white
    rgb = np.zeros((res_y, res_x, 3), dtype=np.float32)
    # Red channel
    rgb[:, :, 0] = np.power(norm_img, 0.45)
    # Green channel
    rgb[:, :, 1] = np.power(norm_img, 0.65) * 0.88 + np.power(norm_img, 2.5) * 0.12
    # Blue channel
    rgb[:, :, 2] = np.power(norm_img, 1.1) * 0.68 + np.power(norm_img, 4.0) * 0.32
    rgb = np.clip(rgb, 0.0, 1.0)
    
    plt.figure(figsize=(15, 8.5), dpi=180)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title("MorphoCalc Mark-IV: Relativistic Ray-Traced Gargantua (Interstellar Optical Match)\n" +
              "Dual Lensed Accretion Halos, Keplerian Striations, and Razor-Thin Einstein Photon Ring",
              fontsize=12, color='white', pad=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05, facecolor='black')
    plt.close()
    print(f"[+] Saved Interstellar-matched render to: {output_path}")

if __name__ == "__main__":
    render_interstellar_gargantua(res_y=360, res_x=640, max_steps=220, dt=0.08)
