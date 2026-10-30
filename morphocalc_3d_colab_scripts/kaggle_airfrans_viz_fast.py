import matplotlib.pyplot as plt
import torch

print("Generating predictions from trained AI...")
with torch.no_grad():
    preds = model(X_train.to(DEVICE)).cpu().numpy()

# DOWNSAMPLE: Only take every 10th point so matplotlib doesn't crash!
skip = 10 
U_pred = preds[::skip, 0] 
X_fast = X[::skip]
Y_fast = Y[::skip]

U_pred_raw = (U_pred * targets_std[0]) + targets_mean[0]
U_real_raw = (Y_train[::skip, 0].numpy() * targets_std[0]) + targets_mean[0]

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_fast, Y_fast, c=U_real_raw, cmap='jet', s=2)
plt.colorbar(label='Wind Velocity (U)')
plt.title('REAL Supercomputer Data (AirfRANS)')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(X_fast, Y_fast, c=U_pred_raw, cmap='jet', s=2)
plt.colorbar(label='Wind Velocity (U)')
plt.title('MorphoCalc AI Prediction')
plt.axis('equal')

plt.tight_layout()
plt.show()
