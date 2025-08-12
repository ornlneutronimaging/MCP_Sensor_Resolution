import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-10, 10)
ax.set_ylim(0, 30)
ax.set_aspect('equal')

# Drawing objects
object_pos = 5  # Initial object position
sensor_pos = 20  # Fixed sensor position
num_rays = 8  # Number of neutron rays to illustrate

object_line, = plt.plot([], [], 'k-', lw=3, label="Object")
sensor_line, = plt.plot([], [], 'r-', lw=3, label="Sensor")
rays_low_div = [plt.plot([], [], 'b-', lw=1)[0] for _ in range(num_rays)]  # Low-divergence beam
rays_high_div = [plt.plot([], [], 'r-', lw=1)[0] for _ in range(num_rays)]  # High-divergence beam

# Initialize animation frame
def init():
    object_line.set_data([], [])
    sensor_line.set_data([], [])
    for ray in rays_low_div + rays_high_div:
        ray.set_data([], [])
    return [object_line, sensor_line] + rays_low_div + rays_high_div

# Update animation frame
def update(frame):
    global object_pos
    object_line.set_data([-1, 1], [object_pos, object_pos])  # Object placement
    sensor_line.set_data([-1, 1], [sensor_pos, sensor_pos])  # Sensor placement
    
    # Update each ray for low-divergence beam
    divergence_low = np.linspace(-0.5, 0.5, num_rays)  # Minimal angular spread
    for i, ray in enumerate(rays_low_div):
        ray.set_data([0, divergence_low[i]*(sensor_pos - object_pos)],
                     [object_pos, sensor_pos])
    
    # Update each ray for high-divergence beam
    divergence_high = np.linspace(-1.5, 1.5, num_rays)  # Larger angular spread
    for i, ray in enumerate(rays_high_div):
        ray.set_data([0, divergence_high[i]*(sensor_pos - object_pos)],
                     [object_pos, sensor_pos])
                     
    object_pos += 0.15  # Gradually move the object farther from the sensor
    return [object_line, sensor_line] + rays_low_div + rays_high_div

# Animate and save as GIF
ani = FuncAnimation(fig, update, init_func=init, frames=100, blit=True, interval=100)
ani.save("neutron_beam_side_profile.gif", dpi=100, writer=PillowWriter())