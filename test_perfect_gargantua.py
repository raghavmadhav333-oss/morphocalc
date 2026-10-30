"""
Test Perfect Gargantua Ray-Tracer (Exact Look-At Matrix & Optically Thick Disk)
Lead Researcher: Raghav
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def render_test(res_y=180, res_x=320, max_steps=180, dt=0.08):
    cam_dist = 18.0
    theta_cam = np.radians(8.0) # 8 deg elevation
    fov = 0.45
    aspect = res_x / res_y

    y_screen, x_screen = torch.meshgrid(
        torch.linspace(-fov, fov, res_y, device=device),
        torch.linspace(-fov * aspect, fov * aspect, res_x, device=device),
        indexing='ij'
    )

    cam_pos = torch.tensor([cam_dist * np.cos(theta_cam), 0.0, cam_dist * np.sin(theta_cam)], device=device, dtype=torch.float32)
    target = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    forward = (target - cam_pos) / torch.norm(target - cam_pos)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    right = torch.linalg.cross(forward, world_up)
    right = right / torch.norm(right)
    up = torch.linalg.cross(right, forward)
    up = up / torch.norm(up)

    # Broadcast to rays
    forward_ray = forward.view(1, 1, 3)
    right_ray = right.view(1, 1, 3)
    up_ray = up.view(1, 1, 3)

    ray_dir = forward_ray + x_screen.unsqueeze(-1) * right_ray + y_screen.unsqueeze(-1) * up_ray
    ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)

    N_rays = res_y * res_x
    p = cam_pos.unsqueeze(0).expand(N_rays, 3).clone()
    v = ray_dir.reshape(N_rays, 3).clone()

    intensity = torch.zeros(N_rays, device=device)
    active = torch.ones(N_rays, dtype=torch.bool, device=device)
    prev_z = p[:, 2].clone()

    M = 1.0
    r_isco = 4.5 * M
    r_out = 15.0 * M

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

        in_disk = (r_cross >= r_isco) & (r_cross <= r_out)
        if in_disk.any():
            disk_idx = crossed_idx[in_disk]
            r_hit = r_cross[in_disk]
            phi_hit = torch.atan2(p[disk_idx, 1], p[disk_idx, 0])

            base_lum = (r_hit / r_isco)**(-2.0)
            filaments = 1.0 + 0.3 * torch.sin(24.0 * torch.log(r_hit) + 4.0 * phi_hit)
            ring = 2.5 * torch.exp(-((r_hit - 3.0 * M) / 0.15)**2)
            
            # Doppler
            v_orb = torch.sqrt(torch.tensor(M, device=device) / r_hit)
            sin_phi = -p[disk_idx, 1] / r_hit
            doppler = (1.0 + 0.7 * v_orb * sin_phi)**2.5

            intensity[disk_idx] += (base_lum * filaments + ring) * doppler
            # Optically thick disk: absorbs ray!
            active[disk_idx] = False

        captured = r < 2.05 * M
        escaped = r > 32.0 * M
        active[curr_idx[captured | escaped]] = False
        prev_z[curr_idx] = curr_z

    img = intensity.reshape(res_y, res_x).cpu().numpy()
    p99 = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0
    norm_img = np.clip(img / (p99 + 1e-9), 0.0, 1.0)

    # 3-channel RGB Interstellar palette
    rgb = np.zeros((res_y, res_x, 3), dtype=np.float32)
    rgb[:, :, 0] = np.power(norm_img, 0.45)
    rgb[:, :, 1] = np.power(norm_img, 0.7) * 0.9 + np.power(norm_img, 2.5) * 0.1
    rgb[:, :, 2] = np.power(norm_img, 1.2) * 0.7 + np.power(norm_img, 4.0) * 0.3
    rgb = np.clip(rgb, 0.0, 1.0)

    plt.figure(figsize=(10, 6), dpi=120)
    plt.imshow(rgb)
    plt.axis('off')
    plt.savefig("d:/Research/morphocalc/test_gargantua_centered.png", bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()
    print("[+] Test frame saved to d:/Research/morphocalc/test_gargantua_centered.png")

if __name__ == "__main__":
    render_test()
