import matplotlib.pyplot as plt
import torch

# 1. Ask the AI to predict the velocities for the entire airfoil point cloud
print("Generating predictions from trained AI...")
with torch.no_grad():
    preds = model(X_train.to(DEVICE)).cpu().numpy()

U_pred = preds[:, 0]  # Predicted X-velocity
P_pred = preds[:, 3]  # Predicted Pressure

# Un-normalize them back to real-world physics values
U_pred_raw = (U_pred * targets_std[0]) + targets_mean[0]
U_real_raw = (Y_train[:, 0].numpy() * targets_std[0]) + targets_mean[0]

# 2. Draw the comparison!
plt.figure(figsize=(15, 6))

# PLOT A: What the Supercomputer says (Ground Truth)
plt.subplot(1, 2, 1)
plt.scatter(X, Y, c=U_real_raw, cmap='jet', s=1)
plt.colorbar(label='Wind Velocity (U)')
plt.title('REAL Supercomputer Data (AirfRANS)')
plt.axis('equal')

# PLOT B: What MorphoCalc AI predicts
plt.subplot(1, 2, 2)
plt.scatter(X, Y, c=U_pred_raw, cmap='jet', s=1)
plt.colorbar(label='Wind Velocity (U)')
plt.title('MorphoCalc AI Prediction')
plt.axis('equal')

plt.tight_layout()
plt.show()
