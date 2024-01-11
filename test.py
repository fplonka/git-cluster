import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example data: m time steps, n points
m, n = 100, 50
data = np.random.rand(m, n, 2)  # Replace with your data

fig, ax = plt.subplots()
points, = plt.plot([], [], 'bo')


def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return points,


def update(frame):
    points.set_data(data[frame, :, 0], data[frame, :, 1])
    return points,


ani = FuncAnimation(fig, update, frames=m, init_func=init, blit=True)

plt.show()
# To save the animation:
# ani.save('animation.mp4')
