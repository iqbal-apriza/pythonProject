import matplotlib.pyplot as plt
import time
import random

# Initialize interactive plot
plt.ion()
fig, ax = plt.subplots()
dot, = ax.plot([], [], 'ro')  # red dot

# Set fixed 2D space
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('2D Real-Time Position Tracker')

# Simulated object position
x = 0
y = 0

while True:
    # Simulate movement (replace this with your own sensor input)
    x += random.uniform(-0.5, 0.5)
    y += random.uniform(-0.5, 0.5)

    # Update dot position
    dot.set_data(x, y)
    plt.draw()
    plt.pause(0.05)  # 20 updates per second
