import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Gradient descent parameters
y_true = 0.5
eta = 0.1
steps = 20
x_init = np.array([-6, -4, -2, 0, 2, 4, 6], dtype=float)

# Store trajectory
trajectories = []

for x0 in x_init:
    x = x0
    traj = [x]
    for _ in range(steps):
        grad = 2 * (sigmoid(x) - y_true) * sigmoid_derivative(x)
        x -= eta * grad
        traj.append(x)
    trajectories.append(traj)

# Plot trajectories
plt.figure(figsize=(10,6))
for i, traj in enumerate(trajectories):
    plt.plot(traj, label=f"x0={x_init[i]}")
plt.xlabel("Step")
plt.ylabel("x value")
plt.title("Gradient Descent Trajectories on Sigmoid (Flat Regions Visible)")
plt.legend()
plt.grid(True)
plt.show()
