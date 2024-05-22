import numpy as np 
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random



L = 100
n = 4000
x_min = 0 
x_max = L
delta_x = (x_max - x_min) / n


# For grass
A = 2.2 # bounded between 0.9 - 2.8
B = 0.45 

# # For trees
# A = 0.08 # bounded between 0.08 - 0.2
# B = 0.045

nu = 182.5

def f_u(u, w):
    # Generate the system of equations in u
    equations_u = [w[i]*u[i]**2 - B*u[i] + (u[i+1] - 2 * u[i] + u[i-1]) / delta_x**2 for i in range(1, n)]
    i = 0
    equations_u.insert(0, w[i]*u[i]**2 - B*u[i] + (u[i+1] - 2 * u[i] + u[-2]) / delta_x**2)  # Equation for i = 0    
    i = n
    equations_u.append( w[i]*u[i]**2 - B*u[i] + (u[i-1] - 2 * u[i] + u[1]) / delta_x**2) # Equation for i = n       
    return np.array(equations_u)

def f_w(u, w):
    # Generate the system of equations in w
    equations_w = [A - w[i] - w[i]*u[i]**2 + nu*(w[i+1] - w[i]) / delta_x for i in range(1, n)]
    i = 0
    equations_w.insert(0, A - w[i] - w[i]*u[i]**2 + nu*(w[i+1] - w[-2])/(2*delta_x))  # Equation for i = 0    
    i = n
    equations_w.append(A - w[i] - w[i]*u[i]**2 + nu*(w[1] - w[i-1])/(2*delta_x)) # Equation for i = n       
    return np.array(equations_w)

def rk4_step(u_n, w_n, dt, f_u, f_w):
    """
    Perform one step of the fourth-order Runge-Kutta method.

    Parameters:
    - u_n, w_n: Current values of u and w at time step n.
    - dt: Time step size.
    - f: Function representing the right-hand side of the equation.

    Returns:
    - u_np1, w_np1: Values of u at the next time step (n+1).
    """

    k1_u = f_u(u_n, w_n)
    k1_w = f_w(u_n, w_n)

    k2_u = f_u(u_n + 0.5 * dt * k1_u, w_n + 0.5 * dt * k1_w)
    k2_w = f_w(u_n + 0.5 * dt * k1_u, w_n + 0.5 * dt * k1_w)
    
    k3_u = f_u(u_n + 0.5 * dt * k2_u, w_n + 0.5 * dt * k2_w)
    k3_w = f_w(u_n + 0.5 * dt * k2_u, w_n + 0.5 * dt * k2_w)

    k4_u = f_u(u_n + dt * k3_u, w_n + dt * k3_w)
    k4_w = f_w(u_n + dt * k3_u, w_n + dt * k3_w)

    u_np1 = u_n + (dt / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
    w_np1 = w_n + (dt / 6.0) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)

    return u_np1, w_np1

def solve_time_dependent_equation(f_u, f_w, u_initial, w_initial, dt, num_steps):
    """
    Solve the time-dependent equation using the fourth-order Runge-Kutta method.

    Parameters:
    - u: Function representing the right-hand side of the equation.
    - w_initial: Initial values of u.
    - dt: Time step size.
    - num_steps: Number of time steps to take.

    Returns:
    - u_values: Array containing the values of u at each time step.
    """

    u_values = [u_initial]
    u_n = u_initial
    
    w_values = [w_initial]
    w_n = w_initial

    for _ in range(num_steps):
        u_np1 = rk4_step(u_n, w_n, dt, f_u, f_w)[0]
        u_values.append(u_np1)
        u_n = u_np1
        
    for _ in range(num_steps):
        w_np1 = rk4_step(u_n, w_n, dt, f_u, f_w)[1]
        w_values.append(w_np1)
        w_n = w_np1
        
    return np.array(u_values), np.array(w_values)


# Initial guess
x = np.linspace(x_min, x_max, n+1)

initial_guess = np.zeros(2*(n + 1))

stable_eq_u = (A + np.sqrt(A**2 - 4*B**2)) / (2*B)
stable_eq_w = (A - np.sqrt(A**2 - 4*B**2)) / (2)

for i in range(0,n+1):
    initial_guess[i] = random.uniform(0.95*stable_eq_u, 1.05*stable_eq_u)
    
for i in range(n+1, 2*(n+1)):
    initial_guess[i] = random.uniform(0.95*stable_eq_w, 1.05*stable_eq_w)

u_initial, w_initial = np.split(initial_guess, 2)


"""plt.figure()
plt.plot(x, u_initial)
plt.plot(x, w_initial)
plt.show()"""



# Time parameters
dt = 1.1*(10**-4) # Time step size
num_steps = 3000  # Number of time steps




# Solve the time-dependent equation
u_values, w_values = solve_time_dependent_equation(f_u, f_w, u_initial, w_initial, dt, num_steps)
 # Plot the evolution of T and G over time on the same plot
"""
def update(frame):
    plt.cla()  # Clear the current axes
    # Plot T and G for the current frame
    plt.plot(x, u_values[frame], label=f'Time Step {frame} (u)', linestyle='-')
    plt.plot(x, w_values[frame], label=f'Time Step {frame} (w)', linestyle='-')
    # Set y-axis limits to ensure they remain fixed
    plt.ylim(0, 16)
    # Customize plot
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title('Evolution of u and w over Time')
    plt.legend(loc = 'right')
 # Create the figure and animate the plot
fig = plt.figure(figsize=(10, 6))
ani = FuncAnimation(fig, update, frames=num_steps+1, interval=100)



# # Show the animation
plt.show()"""
#plt.figure()
plt.clf()
plt.plot(x, u_values[2000], label = 'Time step 2000')
#plt.plot(x, u_values[2003], label = 'Time step 2003')
#plt.plot(x, u_values[2006], label = 'Time step 2006')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Evolution of u')
plt.legend()
plt.show()



"""#plt.figure()
plt.plot(x, w_values[2000], label = 'Time step 2000')
plt.plot(x, w_values[2003], label = 'Time step 2003')
plt.plot(x, w_values[2006], label = 'Time step 2006')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Evolution of w')
plt.legend()
plt.show()
"""
