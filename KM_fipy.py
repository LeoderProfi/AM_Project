import fipy as fp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 100.0  # Length of the domain
nx = 4000  # Number of grid points
dx = L / nx  # Grid spacing
duration = 100.0  # Duration of the simulation
dt = 1.1 * 10**(-4)  # Time step
n = 100  # Update the plot every nth timestep

# Create a 1D periodic mesh
mesh = fp.PeriodicGrid1D(nx=nx, dx=dx)

# Create variables on the mesh
u = fp.CellVariable(name="u", mesh=mesh, value=0.0, hasOld=True)
w = fp.CellVariable(name="w", mesh=mesh, value=0.0, hasOld=True)

# Define parameters
B = 0.45  # Example value for B
v = 182.5  # Example value for v
A = 2.2  # Example value for A

# Define the equations
u.equation = fp.TransientTerm() == w * u**2 - B * u + fp.DiffusionTerm(coeff=1.0)
w.equation = fp.TransientTerm() == A - w - w*u**2 + v*fp.ConvectionTerm(coeff=(1.0,)) 

# Set initial conditions
u.setValue(7.5+7.5*np.sin(10 * np.pi * mesh.cellCenters[0] / L + 2))  # Sinusoidal initial condition for u
w.setValue(0.1+0.1*np.sin(10 * np.pi * mesh.cellCenters[0] / L))  # Sinusoidal initial condition for w
"""
# Set up the plots
plt.ion()  # Turn on interactive mode

fig1, ax1 = plt.subplots()  # Create a figure and an axes for u
line_u, = ax1.plot(mesh.cellCenters[0], u.value, label="u")  # Line for u
ax1.legend()  # Add a legend

fig2, ax2 = plt.subplots()  # Create a figure and an axes for w
line_w, = ax2.plot(mesh.cellCenters[0], w.value, label="w")  # Line for w
ax2.legend()  # Add a legend

# Solve the equations
time = 0.0
counter = 0
while time < duration:
    u.updateOld()
    w.updateOld()
    
    res_u = 1e+10
    res_w = 1e+10
    while res_u > 1e-6 and res_w > 1e-6:
        res_u = u.equation.sweep(var=u, dt=dt)
        res_w = w.equation.sweep(var=w, dt=dt)
    
    # Plot the variables after specific timesteps
    if counter in [2000, 2200, 2400]:
        line_u.set_ydata(u.value)
        line_w.set_ydata(w.value)
        
        ax1.relim()  # Recompute the data limits
        ax1.autoscale_view()  # Rescale the view
        ax2.relim()  # Recompute the data limits
        ax2.autoscale_view()  # Rescale the view
        
        fig1.canvas.draw()  # Redraw the figure
        fig2.canvas.draw()  # Redraw the figure
        
        plt.pause(0.001)  # Pause for a short period
    
    time += dt
    counter += 1

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot

"""

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()  # Create a figure and an axes
line_u, = ax.plot(mesh.cellCenters[0], u.value, label="u")  # Line for u
line_w, = ax.plot(mesh.cellCenters[0], w.value, label="w")  # Line for w
ax.legend()  # Add a legend

# Set up the plots
plt.ion()  # Turn on interactive mode

fig1, ax1 = plt.subplots()  # Create a figure and an axes for u
line_u, = ax1.plot(mesh.cellCenters[0], u.value, label="u")  # Line for u
ax1.legend()  # Add a legend

fig2, ax2 = plt.subplots()  # Create a figure and an axes for w
line_w, = ax2.plot(mesh.cellCenters[0], w.value, label="w")  # Line for w
ax2.legend()  # Add a legend

# Solve the equations
time = 0.0
counter = 0
while time < duration:
    u.updateOld()
    w.updateOld()
    
    res_u = 1e+10
    res_w = 1e+10
    while res_u > 1e-6 and res_w > 1e-6:
        res_u = u.equation.sweep(var=u, dt=dt)
        res_w = w.equation.sweep(var=w, dt=dt)

    
    # Update the plot every nth timestep
    #if counter % n == 0:
    line_u.set_ydata(u.value)
    line_w.set_ydata(w.value)
    
    ax1.relim()  # Recompute the data limits
    ax1.autoscale_view()  # Rescale the view
    ax2.relim()  # Recompute the data limits
    ax2.autoscale_view()  # Rescale the view
    
    fig1.canvas.draw()  # Redraw the figure
    fig2.canvas.draw()  # Redraw the figure
    
    plt.pause(0.001)  # Pause for a short period
    
    time += dt
    counter += 1

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot"""