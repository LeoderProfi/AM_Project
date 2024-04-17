import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L = 10.0             # length of the domain
Nx = 100             # number of spatial points
Dx = L / Nx          # spatial resolution
Dt = 0.0001           # time step
T = 50.0              # total time to simulate
p1, p2 = 0.003, 0.014   # parameters for u
p3, p4, p5, p6 = 0.638, 0.042, 8, 0.276  # parameters for v

# Discretization
x = np.linspace(0, L, Nx)
u = np.zeros(Nx)
v = np.zeros(Nx)
u_new = np.zeros(Nx)
v_new = np.zeros(Nx)

# Initial conditions
u[:] = np.exp(-((x - L/2)**2) / (2*(L/10)**2))  # Gaussian initial condition for u
v[:] = np.exp(-((x - L/3)**2) / (2*(L/12)**2))  # Gaussian initial condition for v

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
line_u, = ax.plot(x, u, label='u(x, t)')
line_v, = ax.plot(x, v, label='v(x, t)')
ax.set_title('Concentration profiles')
ax.set_xlabel('x')
ax.set_ylabel('Concentration')
ax.legend()

def update(frame):
    global u, v, u_new, v_new
    # Time stepping
    for i in range(1, Nx-1):
        u_new[i] = u[i] + Dt * (u[i] * (1 - u[i]) - p1 * u[i] + (u[i+1] - 2*u[i] + u[i-1]) / Dx**2 - p2 * u[i] * v[i])
        v_new[i] = v[i] + Dt * (p3 * v[i] * (1 - v[i]) - p4 * v[i] + p5 * (v[i+1] - 2*v[i] + v[i-1]) / Dx**2 - p6 * u[i] * v[i])

    # Neumann boundary conditions (zero flux)
    u_new[0] = u[0] + Dt * (u[0] * (1 - u[0]) - p1 * u[0] + (u[1] - 2*u[0] + u[1]) / Dx**2 - p2 * u[0] * v[0])
    u_new[-1] = u[-1] + Dt * (u[-1] * (1 - u[-1]) - p1 * u[-1] + (u[-2] - 2*u[-1] + u[-2]) / Dx**2 - p2 * u[-1] * v[-1])
    v_new[0] = v[0] + Dt * (p3 * v[0] * (1 - v[0]) - p4 * v[0] + p5 * (v[1] - 2*v[0] + v[1]) / Dx**2 - p6 * u[0] * v[0])
    v_new[-1] = v[-1] + Dt * (p3 * v[-1] * (1 - v[-1]) - p4 * v[-1] + p5 * (v[-2] - 2*v[-1] + v[-2]) / Dx**2 - p6 * u[-1] * v[-1])

    # Update current values
    u = np.copy(u_new)
    v = np.copy(v_new)

    # Update the line data
    line_u.set_ydata(u)
    line_v.set_ydata(v)

    return line_u, line_v,

ani = FuncAnimation(fig, update, frames=int(10*T/Dt), blit=True)

plt.show()