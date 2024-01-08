import numpy as np
import matplotlib.pyplot as plt


def sgdr_schedule(t, lr_max, lr_min, T_cur, T_i):
    """
    Calculate the learning rate at a given iteration using SGDR.

    :param t: Current iteration.
    :param lr_max: Maximum learning rate (start of cycle).
    :param lr_min: Minimum learning rate (end of cycle).
    :param T_cur: Current number of iterations since the last restart.
    :param T_i: Number of iterations in the current cycle.
    :return: Learning rate for iteration t.
    """
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * T_cur / T_i))


# Example usage
lr_max = 0.1  # Maximum learning rate
lr_min = 0.0001  # Minimum learning rate
T_0 = 1000  # Initial number of iterations in the first cycle
T_mult = 2  # Factor by which to increase the number of iterations in each cycle

learning_rates = []
t = 0
T_cur = 0
T_i = T_0

while t < 511000:  # Total number of iterations
    lr = sgdr_schedule(t, lr_max, lr_min, T_cur, T_i)
    learning_rates.append(lr)

    t += 1
    T_cur += 1

    if T_cur == T_i:
        # End of current cycle, prepare for the next cycle
        T_cur = 0
        T_i *= T_mult  # Increase the length of the next cycle

# Plotting the learning rates
plt.figure(figsize=(10, 6))
plt.plot(learning_rates)
plt.title("SGDR Learning Rate Schedule")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()
