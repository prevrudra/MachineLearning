import numpy as np
# Example logits (predictions before softmax)
logits = np.array([1000, 1001, 1002])  # large numbers
y_true = np.array([0, 0, 1])           # true class is index 2

# Naive softmax
# Shift logits by max value
shifted_logits = logits - np.max(logits)  # prevents overflow
exp_shifted = np.exp(shifted_logits)
softmax_stable = exp_shifted / np.sum(exp_shifted)


loss_stable = -np.sum(y_true * np.log(softmax_stable))

print("Stable softmax loss:", loss_stable)

