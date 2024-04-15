import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
grid_size = (50, 50)  # Size of the 2D grid
tree_growth_probability = 0.2  # Probability of a tree growing in an empty spot next to a tree
fire_probability = 0.02  # Base probability of a tree catching fire
max_time_steps = 100  # Number of time steps to simulate

# Initialize the grid randomly
np.random.seed(42)
grid = np.random.choice([0, 1], size=grid_size, p=[0.8, 0.2])  # 0 is empty, 1 is tree

def update_grid(grid):
    new_grid = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = grid[max(i-1, 0):min(i+2, grid.shape[0]), max(j-1, 0):min(j+2, grid.shape[1])]
            tree_count = np.sum(neighbors) - grid[i, j]  # Exclude the center cell
            if grid[i, j] == 0 and tree_count > 0:
                if np.random.rand() < tree_growth_probability * tree_count:
                    new_grid[i, j] = 1
            elif grid[i, j] == 1:
                fire_risk = fire_probability + 0.05 * tree_count
                if np.random.rand() < fire_risk:
                    new_grid[i, j] = 0
    return new_grid

# Prepare for visualization
fig, ax = plt.subplots(figsize=(8, 8))
plt.title('2D Simulation of Forestation in Savanna')
im = ax.imshow(grid, cmap='Greens', interpolation='nearest')

def animate(t):
    global grid
    grid = update_grid(grid)
    im.set_data(grid)
    ax.set_title(f'Time step: {t + 1}')
    return im,

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=max_time_steps, repeat=False)

plt.show()
