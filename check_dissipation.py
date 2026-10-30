import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import scipy.io as sio
import os

dest = os.path.join(os.environ.get('TEMP', '/tmp'), "cylinder_nektar_wake.mat")
mat = sio.loadmat(dest)
X_star = mat['X_star']
U_star = mat['U_star']
u_all = U_star[:, 0, :]
v_all = U_star[:, 1, :]
u_mean = np.mean(u_all, axis=1)
v_mean = np.mean(v_all, axis=1)
u_prime = u_all - u_mean[:, None]
v_prime = v_all - v_mean[:, None]
tau_12_real = np.mean(u_prime * v_prime, axis=1)

u_grid = u_mean.reshape(50, 100)
v_grid = v_mean.reshape(50, 100)
x_u = np.unique(X_star[:, 0])
y_u = np.unique(X_star[:, 1])
dx = float(x_u[1] - x_u[0])
dy = float(y_u[1] - y_u[0])
u_y, u_x = np.gradient(u_grid, dy, dx)
v_y, v_x = np.gradient(v_grid, dy, dx)

S_12 = 0.5 * (u_y.flatten() + v_x.flatten())
S_mag = np.sqrt(2.0 * (u_x.flatten()**2 + v_y.flatten()**2 + 2.0 * S_12**2))
Omega_12 = 0.5 * (v_x.flatten() - u_y.flatten())
tau_std = np.std(tau_12_real) + 1e-8
tau_norm = tau_12_real / tau_std

x_coords = X_star[:, 0]
train_mask = (x_coords <= 4.5)
test_mask = (x_coords > 4.5)

Delta = 0.1
c1 = -Delta**2 * S_mag * S_12
c2 = Delta**2 * Omega_12 * u_x.flatten()
c3 = Delta**2 * S_12
c4 = -Delta**2 * S_12 * u_x.flatten()

A_train = np.stack([c1[train_mask], c2[train_mask], c3[train_mask], c4[train_mask]], axis=1)
y_train = tau_norm[train_mask]
A_test = np.stack([c1[test_mask], c2[test_mask], c3[test_mask], c4[test_mask]], axis=1)
y_test = tau_norm[test_mask]

coeffs, _, _, _ = np.linalg.lstsq(A_train, y_train, rcond=1e-4)
pred_test_morpho = A_test @ coeffs

# Notice the two signs:
diss_plus = pred_test_morpho * S_12[test_mask]
diss_minus = -pred_test_morpho * S_12[test_mask]

print(f"Positive Dissipation with +tau*S : {np.mean(diss_plus >= -1e-6) * 100.0:.1f}%")
print(f"Positive Dissipation with -tau*S : {np.mean(diss_minus >= -1e-6) * 100.0:.1f}%")
