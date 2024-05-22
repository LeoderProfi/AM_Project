import numpy as np 
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter


def fire(dimension, n_cells, delta, Solution, fire_starting_point, fire_radius):
    #check dimensions:
    if dimension == 1:
        fire_starting_point = fire_starting_point[0] #Convert the starting point to a scalar
        ST, SG = np.split(Solution, 2) #Split the solution into the two components
        fire_starting_index_x = int(fire_starting_point / delta) #Find the index of the starting point of the fire
        X = np.arange(n_cells+1) #Create an array of the x-values
        distance = np.abs(X - fire_starting_index_x) #Calculate the distance from the starting point in which the fire should be active
        mask = distance <= fire_radius/delta #Create a mask for the cells that should be affected by the fire
        ST[mask] = 0.5 * ST[mask] #Set the Tree concentration of the cells affected by the fire to 0
       # SG[mask] = 0.0 #Set the gras concentration of the cells affected by the fire to 0
        return np.concatenate((ST, SG))
    elif dimension == 2:
        n_x = n_cells
        n_y = n_cells
        dx, dy = delta
        ST, SG = np.split(Solution, 2)
        T_reshaped = np.reshape(ST, (n_x+1,n_y+1))
        G_reshaped = np.reshape(SG, (n_x+1,n_y+1))
        fire_starting_index_x = int(fire_starting_point[0] / dx)
        fire_starting_index_y = int(fire_starting_point[1] / dy)
        X, Y = np.meshgrid(np.arange(n_x+1), np.arange(n_y+1))
        distance = np.hypot(X - fire_starting_index_x, Y - fire_starting_index_y)
        mask = np.logical_and(distance <= fire_radius/dx, distance >= 0)
        T_reshaped[mask] = 0.5 * T_reshaped[mask]
        G_reshaped[mask] = 0
        return np.concatenate((T_reshaped.flatten(), G_reshaped.flatten()))
    else:
        print("Error: Something went wrong with the dimensions in the fire function.")

def Initialisation1D():
    dimension = 1
    mu = 1
    Lambda = 1

    L = 10000
    n = 100
    x_min = 0
    x_max = L

    dt = 0.01
    num_steps = 100
    delta = (x_max - x_min) / n

    return dimension, L, n, x_min, x_max, delta, mu, Lambda, dt, num_steps

def Initialisation2D():
    dimension = 2
    mu = 1
    Lambda = 1

    L = 1000
    n =40
    
    dt = 0.0001
    num_steps = 100

    xy_min = [0, 0]
    xy_max = [L, L]

    delta = np.zeros(dimension)
    delta[0] = (xy_max[0] - xy_min[0]) / n
    delta[1] = (xy_max[1] - xy_min[1]) / n
    return dimension, L, n, xy_min, xy_max, delta, mu, Lambda, dt, num_steps

def solve_time_dependent_equation(T_initial, G_initial, dt, num_steps, mu, Lambda, L, delta, n):
    def f_T(T, mu, Lambda, L, delta, n):
        delta_x = delta
        # Generate the system of equations in T (RHO IN PAPER)
        equations_T = [(1-mu)*T[i] + (Lambda -1)* T[i]**2 - T[i]**3 + (1/2)*(L**2-T[i])* (T[i+1]-T[i-1])/(2*delta_x) - (1/8)*T[i]*(T[i+1] - 2 * T[i] + T[i-1]) / delta_x**2 for i in range(1, n)]
        equations_T.insert(0, (1-mu)*T[0] + (Lambda -1)* T[0]**2 - T[0]**3 - (1/8)*T[0]*(2*T[1] - 2 * T[0]) / delta_x**2)  # Equation for i = 0    #NEUMANN BC
        equations_T.append((1-mu)*T[-1] + (Lambda -1)* T[-1]**2 - T[-1]**3 - (1/8)*T[-1]*(2*T[-2] - 2 * T[-1]) / delta_x**2)  # Equation for i = n      
        return np.array(equations_T)

    def rk4_step(T_n, dt, f_T, mu, Lambda, L, delta, n):
        """
        Perform one step of the fourth-order Runge-Kutta method.

        Parameters:
        - T_n, G_n: Current values of T and G at time step n.
        - dt: Time step size.
        - f: Function representing the right-hand side of the equation.

        Returns:
        - T_np1, G_np1: Values of T at the next time step (n+1).
        """

        k1_T = f_T(T_n, mu, Lambda, L, delta, n)

        k2_T = f_T(T_n + 0.5 * dt * k1_T, mu, Lambda, L, delta, n)
        
        k3_T = f_T(T_n + 0.5 * dt * k2_T, mu, Lambda, L, delta, n)

        k4_T = f_T(T_n + dt * k3_T, mu, Lambda, L, delta, n)

        T_np1 = T_n + (dt / 6.0) * (k1_T + 2 * k2_T + 2 * k3_T + k4_T)
        return T_np1

    """
    Solve the time-dependent equation using the fourth-order Runge-Kutta method.

    Parameters:
    - F: Function representing the right-hand side of the equation.
    - T_initial: Initial values of T.
    - dt: Time step size.
    - num_steps: Number of time steps to take.

    Returns:
    - T_values: Array containing the values of T at each time step.
    """
    T_values = [T_initial]
    T_n = T_initial

    for _ in range(num_steps):
        T_np1 = rk4_step(T_n, dt, f_T, mu, Lambda, L, delta, n)
        T_values.append(T_np1)
        T_n = T_np1

    return np.array(T_values)

def f(T, mu, Lambda, L, delta, n): ### NOTE THAT THIS IS THE SAME AS F_T, CAN BE WRITTEN DOWN MORE EFFICIENTLY
        delta_x = delta
        # Generate the system of equations in T (RHO IN PAPER)
        equations_T = [(1-mu)*T[i] + (Lambda -1)* T[i]**2 - T[i]**3 + (1/2)*(L**2-T[i])* (T[i+1]-T[i-1])/(2*delta_x) - (1/8)*T[i]*(T[i+1] - 2 * T[i] + T[i-1]) / delta_x**2 for i in range(1, n)]
        equations_T.insert(0, (1-mu)*T[0] + (Lambda -1)* T[0]**2 - T[0]**3 - (1/8)*T[0]*(2*T[1] - 2 * T[0]) / delta_x**2)  # Equation for i = 0    #NEUMANN BC
        equations_T.append((1-mu)*T[-1] + (Lambda -1)* T[-1]**2 - T[-1]**3 - (1/8)*T[-1]*(2*T[-2] - 2 * T[-1]) / delta_x**2)  # Equation for i = n      
        return np.array(equations_T)

def J(T, mu, Lambda, L, delta, n):
    delta_x = delta
    jac_matrix = np.zeros((n+1, n+1))

    for i in range(1, n):
            jac_matrix[i, i] = 1-mu + 2*(Lambda -1)*T[i] - 3*T[i]**2  / delta_x**2  - (1/2)*T[i]* (T[i+1]-T[i-1])/(2*delta_x) - (1/8)*(T[i+1] + T[i-1] - 4*T[i]) / delta_x**2
            jac_matrix[i, i-1] =  - (1/2)*(L**2-T[i])/(2*delta_x) - (1/8)*T[i]/ delta_x**2
            jac_matrix[i, i+1] = (1/2)*(L**2-T[i])/(2*delta_x) - (1/8)*T[i]/ delta_x**2

    # Jacobian for i = 0
    i = 0
    jac_matrix[i, i+1] = - 2*(1/8)*T[i]/ delta_x**2
    jac_matrix[i, i] = 1-mu + 2*(Lambda -1)*T[i] - 3*T[i]**2  / delta_x**2  - (1/8)*(2* T[i+1] - 4*T[i]) / delta_x**2
    
    # Jacobian for i = n
    i = n
    jac_matrix[i, i-1] = - 2*(1/8)*T[i]/ delta_x**2
    jac_matrix[i, i] = 1-mu + 2*(Lambda -1)*T[i] - 3*T[i]**2  / delta_x**2  - (1/8)*(2* T[i-1] - 4*T[i]) / delta_x**2

    return jac_matrix

def newton_raphson_system(f, J, mu, Lambda, L, delta, n, initial_guess, tol=1e-6, max_iter=10000):
    for i in range(max_iter):
        T = initial_guess
        # Evaluate the system of equations and the Jacobian at the current point
        f_val = np.array(f(T, mu, Lambda, L, delta, n))
        J_val = np.array(J(T, mu, Lambda, L, delta, n))

        # Solve the linear system to get the update
        delta_T_G = spsolve(J_val, -f_val)
       
        # Update the solution
        initial_guess += delta_T_G
        #print(delta_T_G)
        # Check for convergence
        if np.linalg.norm(delta_T_G) < tol:
            print('Converged after', i + 1, 'iterations.')
            return initial_guess  # Return the solution
   
    raise RuntimeError("Newton-Raphson method did not converge within the maximum number of iterations.")

def initial_guess_func(x_min, x_max, n):
    # Initial guess
    x = np.linspace(x_min, x_max, n+1)

    initial_guess = np.zeros(2*(n + 1))
    for i in range(0,n+1):
        initial_guess[i] = 0.909*np.cos(i)
    
    for i in range(n+1, 2*(n+1)):
        initial_guess[i] = 0.538*np.cos(i)
    return initial_guess

def initial_guess_func1(x_min, x_max, n):
    # Initial guess
    x = np.linspace(x_min, x_max, n+1)

    initial_guess = np.zeros(2*(n + 1))
    for i in range(int(n/2)-1,int(n/2)+1):
        initial_guess[i] = 0.9
    
    for i in range(n+1 + int(n/2)-3,n+1+ int(n/2)-1):
        initial_guess[i] = 0.9
    return initial_guess
   
def plot_solution1D(T_solution, n, x_min, x_max, moment, save = False):
    x = np.linspace(x_min, x_max, n+1)
    plt.figure()
    plt.plot(x, T_solution, label='T')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("./Pictures/1D_{}.pdf".format(moment))
    plt.clf()

    #plt.show()

def plot_solution_2D(T_solution, n, moment, save = False):
    x_vertices = (n+1)
    y_vertices = (n+1)

    T_reshaped = np.reshape(T_solution, (x_vertices,y_vertices))

    plt.figure()
    plt.imshow(T_reshaped, extent = [0,1,1,0])
    plt.colorbar()
    plt.ylabel('y-axis')
    plt.xlabel('x-axis')
    plt.tight_layout()
    if save:
        plt.savefig("./Pictures/2D_{}_1.pdf".format(moment))
    plt.clf()

    #plt.show()



##### I HAVE CHANGED IT UNTILL HERE, SO I DID THE 1D CASE #####

def solve_time_dependent_equation2D(T_initial, G_initial, p, delta, n, alpha_1, alpha_2, dt, num_steps):
    def f_T2D(T, G, p, delta, n, alpha_1, alpha_2):
        delta_x = delta[0]
        delta_y = delta[1]
        p_1, p_2, p_3, p_4, p_5, p_6 = p
        equations_T = np.zeros((n+1)*(n+1))
        for i in range((n+1)*(n+1)):
            if i == 0 : #top left corner
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i+1] + 2*T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            elif i == n+1: #top right corner
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i-1] + 2*T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            elif i == n*(n+1): #bottom left corner
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i+1] + 2*T[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            elif i == n*(n+1) + n: #bottom right corner
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i-1] + 2*T[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            elif i < n+1: #upper boundary
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(T[i+1] + T[i-1] + 2*T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            elif i % (n+1) == 0: #left boundary
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i+1] +  T[i-(n+1)] + T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            elif i % (n+1) == n: #right boundary
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i-1] + T[i-(n+1)] + T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            elif i > n*(n+1): #lower boundary
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(T[i+1] + T[i-1] + 2*T[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            else:
                equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(T[i+1] + T[i-1] + T[i-(n+1)] + T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]

        return np.array(equations_T)

    def f_G2D(T, G, p, delta, n, alpha_1, alpha_2):
        delta_x = delta[0]
        delta_y = delta[1]
        p_1, p_2, p_3, p_4, p_5, p_6 = p
        equations_G = np.zeros((n+1)*(n+1))
        for i in range((n+1)*(n+1)):
            if i == 0 : #top left corner
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (2*G[i+1] + 2*G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
            elif i == n+1: #top right corner
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (2*G[i-1] + 2*G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
            elif i == n*(n+1): #bottom left corner
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (2*G[i+1] + 2*G[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
            elif i == n*(n+1) + n: #bottom right corner
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (2*G[i-1] + 2*G[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
            elif i < n+1: #upper boundary
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (G[i+1] + G[i-1] + 2*G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
            elif i % (n+1) == 0: #left boundary
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (2*G[i+1] + G[i-(n+1)] + G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
            elif i % (n+1) == n: #right boundary
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (2*G[i-1] + G[i-(n+1)] + G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
            elif i > n*(n+1): #lower boundary
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (G[i+1] + G[i-1] + 2*G[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
            else:
                equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + alpha_2 * (G[i+1] + G[i-1] + G[i-(n+1)] + G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
    
        return np.array(equations_G)

    def rk4_step2D(T_n, G_n, dt, f_T, f_G, p, delta, n, alpha_1, alpha_2):
        """
        Perform one step of the fourth-order Runge-Kutta method.

        Parameters:
        - T_n, G_n: Current values of T and G at time step n.
        - dt: Time step size.
        - f: Function representing the right-hand side of the equation.

        Returns:
        - T_np1, G_np1: Values of T at the next time step (n+1).
        """

        k1_T = f_T(T_n, G_n, p, delta, n, alpha_1, alpha_2)
        k1_G = f_G(T_n, G_n, p, delta, n, alpha_1, alpha_2)

        k2_T = f_T(T_n + 0.5 * dt * k1_T, G_n + 0.5 * dt * k1_G, p, delta, n, alpha_1, alpha_2)
        k2_G = f_G(T_n + 0.5 * dt * k1_T, G_n + 0.5 * dt * k1_G, p, delta, n, alpha_1, alpha_2)
    
        k3_T = f_T(T_n + 0.5 * dt * k2_T, G_n + 0.5 * dt * k2_G, p, delta, n, alpha_1, alpha_2)
        k3_G = f_G(T_n + 0.5 * dt * k2_T, G_n + 0.5 * dt * k2_G, p, delta, n, alpha_1, alpha_2)

        k4_T = f_T(T_n + dt * k3_T, G_n + dt * k3_G, p, delta, n, alpha_1, alpha_2)
        k4_G = f_G(T_n + dt * k3_T, G_n + dt * k3_G, p, delta, n, alpha_1, alpha_2)



        T_np1 = T_n + (dt / 6.0) * (k1_T + 2 * k2_T + 2 * k3_T + k4_T)
        G_np1 = G_n + (dt / 6.0) * (k1_G + 2 * k2_G + 2 * k3_G + k4_G)

        return T_np1, G_np1

    """
    Solve the time-dependent equation using the fourth-order Runge-Kutta method.

    Parameters:
    - F: Function representing the right-hand side of the equation.
    - T_initial: Initial values of T.
    - dt: Time step size.
    - num_steps: Number of time steps to take.

    Returns:
    - T_values: Array containing the values of T at each time step.
    """

    T_values = [T_initial]
    T_n = T_initial
   
    G_values = [T_initial]
    G_n = G_initial

    for _ in range(num_steps):
        T_np1 = rk4_step2D(T_n, G_n, dt, f_T2D, f_G2D, p, delta, n, alpha_1, alpha_2)[0]
        T_values.append(T_np1)
        T_n = T_np1
       
       

    for _ in range(num_steps):
        G_np1 = rk4_step2D(T_n, G_n, dt, f_T2D, f_G2D, p, delta, n, alpha_1, alpha_2)[1]
        G_values.append(G_np1)
        G_n = G_np1
       

    return np.array(T_values), np.array(G_values)

def f_2D(T, G, p, delta, n, alpha_1, alpha_2):
    p_1, p_2, p_3, p_4, p_5, p_6 = p
    delta_x = delta[0]
    equations_T = np.zeros((n+1)*(n+1))
    equations_G = np.zeros((n+1)*(n+1))
    for i in range((n+1)*(n+1)):
        if i == 0 : #top left corner
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i+1] + 2*T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (2*G[i+1] + 2*G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
        elif i == n+1: #top right corner
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i-1] + 2*T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (2*G[i-1] + 2*G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
        elif i == n*(n+1): #bottom left corner
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i+1] + 2*T[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (2*G[i+1] + 2*G[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
        elif i == n*(n+1) + n: #bottom right corner
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i-1] + 2*T[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (2*G[i-1] + 2*G[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
        elif i < n+1: #upper boundary
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(T[i+1] + T[i-1] + 2*T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (G[i+1] + G[i-1] + 2*G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
        elif i % (n+1) == 0: #left boundary
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i+1] +  T[i-(n+1)] + T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (2*G[i+1] + G[i-(n+1)] + G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
        elif i % (n+1) == n: #right boundary
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(2*T[i-1] + T[i-(n+1)] + T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (2*G[i-1] + G[i-(n+1)] + G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
        elif i > n*(n+1): #lower boundary
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(T[i+1] + T[i-1] + 2*T[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (G[i+1] + G[i-1] + 2*G[i-(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
        else:
            equations_T[i] = T[i] * (1 - T[i]) - p_1 * T[i] + alpha_1*(T[i+1] + T[i-1] + T[i-(n+1)] + T[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_2 * T[i] * G[i]
            equations_G[i] = p_3 * G[i] * (1 - G[i]) - p_4 * G[i] + p_5 * (G[i+1] + G[i-1] + G[i-(n+1)] + G[i+(n+1)] - 4 * T[i]) / delta_x**2 - p_6 * T[i] * G[i]
   
    print(type(equations_T))
    equations_T_lst = list(equations_T)
    equations_G_lst = list(equations_G)
    print(len(equations_T))
    equations_vector = equations_T_lst + equations_G_lst
    return equations_vector

def J_2D(T, G, p, delta, n, alpha_1, alpha_2):

    def J_eq_1_T(T, G, p, delta, n, alpha_1, alpha_2):
        p_1, p_2, p_3, p_4, p_5, p_6 = p
        delta_x = delta[0]
        # Generate the Jacobian matrix
        jac_matrix = np.zeros(((n+1)*(n+1), (n+1)*(n+1)))
        for i in range((n+1)*(n+1)):
            jac_matrix[i, i] = 1 - 2 * T[i] - p_1 - alpha_1*4 / delta_x**2 - p_2 * G[i]
            if i == 0 : #top left corner
                jac_matrix[i, i+1] = alpha_1*2 / delta_x**2
                jac_matrix[i, i+(n+1)] = alpha_1*2 / delta_x**2
            elif i == n+1: #top right corner
                jac_matrix[i, i-1] = alpha_1*2 / delta_x**2
                jac_matrix[i, i+(n+1)] = alpha_1*2 / delta_x**2
            elif i == n*(n+1): #bottom left corner
                jac_matrix[i, i+1] = alpha_1*2 / delta_x**2
                jac_matrix[i, i-(n+1)] = alpha_1*2 / delta_x**2
            elif i == n*(n+1) + n: #bottom right corner
                jac_matrix[i, i-1] = alpha_1*2 / delta_x**2
                jac_matrix[i, i-(n+1)] = alpha_1*2 / delta_x**2
            elif i < n+1: #upper boundary
                jac_matrix[i, i-1] = alpha_1*1 / delta_x**2
                jac_matrix[i, i+1] = alpha_1*1 / delta_x**2
                jac_matrix[i, i+(n+1)] = alpha_1*2 / delta_x**2
            elif i % (n+1) == 0: #left boundary
                jac_matrix[i, i+1] = alpha_1*2 / delta_x**2
                jac_matrix[i, i-(n+1)] = alpha_1*1 / delta_x**2
                jac_matrix[i, i+(n+1)] = alpha_1*1 / delta_x**2
            elif i % (n+1) == n: #right boundary
                jac_matrix[i, i-1] = alpha_1*2 / delta_x**2
                jac_matrix[i, i-(n+1)] = alpha_1*1 / delta_x**2
                jac_matrix[i, i+(n+1)] = alpha_1*1 / delta_x**2
            elif i > n*(n+1): #lower boundary
                jac_matrix[i, i-1] = alpha_1*1 / delta_x**2
                jac_matrix[i, i+1] = alpha_1*1 / delta_x**2
                jac_matrix[i, i-(n+1)] = alpha_1*2 / delta_x**2
            else: #inner points
                jac_matrix[i, i-1] = alpha_1*1 / delta_x**2
                jac_matrix[i, i+1] = alpha_1*1 / delta_x**2
                jac_matrix[i, i-(n+1)] = alpha_1*1 / delta_x**2
                jac_matrix[i, i+(n+1)] = alpha_1*1 / delta_x**2

        return jac_matrix

    def J_eq_1_G(T, G, p, delta, n, alpha_1, alpha_2):
        p_1, p_2, p_3, p_4, p_5, p_6 = p
        delta_x = delta[0]
        # Generate the Jacobian matrix
        jac_matrix = np.zeros(((n+1)*(n+1), (n+1)*(n+1)))
        for i in range(0, (n+1)*(n+1)):
            jac_matrix[i, i] = -p_2 * T[i]
        return jac_matrix

    def J_eq_2_T(T, G, p, delta, n, alpha_1, alpha_2):
        p_1, p_2, p_3, p_4, p_5, p_6 = p
        delta_x = delta[0]
        # Generate the Jacobian matrix
        jac_matrix = np.zeros(((n+1)*(n+1), (n+1)*(n+1)))
        for i in range(0, (n+1)*(n+1)):
            jac_matrix[i, i] = -p_6 * G[i]
        return jac_matrix

    def J_eq_2_G(T, G, p, delta, n, alpha_1, alpha_2):
        p_1, p_2, p_3, p_4, p_5, p_6 = p
        delta_x = delta[0]
        # Generate the Jacobian matrix
        jac_matrix = np.zeros(((n+1)*(n+1), (n+1)*(n+1)))
    
        for i in range((n+1)*(n+1)):
            jac_matrix[i, i] =  p_3 - 2 * p_3 * G[i] - p_4 - 4 * alpha_2 / delta_x**2 - p_6 * T[i]
            if i == 0 : #top left corner
                jac_matrix[i, i+1] = 2*alpha_2 / delta_x**2
                jac_matrix[i, i+(n+1)] = 2*alpha_2 / delta_x**2
            elif i == n+1: #top right corner
                jac_matrix[i, i-1] = 2*alpha_2 / delta_x**2
                jac_matrix[i, i+(n+1)] = 2*alpha_2 / delta_x**2
            elif i == n*(n+1): #bottom left corner
                jac_matrix[i, i+1] = 2*alpha_2 / delta_x**2
                jac_matrix[i, i-(n+1)] = 2*alpha_2 / delta_x**2
            elif i == n*(n+1) + n: #bottom right corner
                jac_matrix[i, i-1] = 2*alpha_2 / delta_x**2
                jac_matrix[i, i-(n+1)] = 2*alpha_2 / delta_x**2
            elif i < n+1: #upper boundary
                jac_matrix[i, i-1] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i+1] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i+(n+1)] = 2*alpha_2 / delta_x**2
            elif i % (n+1) == 0: #left boundary
                jac_matrix[i, i+1] = 2*alpha_2 / delta_x**2
                jac_matrix[i, i-(n+1)] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i+(n+1)] = 1*alpha_2 / delta_x**2
            elif i % (n+1) == n: #right boundary
                jac_matrix[i, i-1] = 2*alpha_2 / delta_x**2
                jac_matrix[i, i-(n+1)] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i+(n+1)] = 1*alpha_2 / delta_x**2
            elif i > n*(n+1): #lower boundary
                jac_matrix[i, i-1] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i+1] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i-(n+1)] = 2*alpha_2 / delta_x**2
            else: #inner points
                jac_matrix[i, i-1] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i+1] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i-(n+1)] = 1*alpha_2 / delta_x**2
                jac_matrix[i, i+(n+1)] = 1*alpha_2 / delta_x**2

        return jac_matrix

    p_1, p_2, p_3, p_4, p_5, p_6 = p
    delta_x = delta[0]
    Jac_eq_1_T = J_eq_1_T(T, G, p, delta, n, alpha_1, alpha_2)
    Jac_eq_1_G = J_eq_1_G(T, G, p, delta, n, alpha_1, alpha_2)
    Jac_eq_2_T = J_eq_2_T(T, G, p, delta, n, alpha_1, alpha_2)
    Jac_eq_2_G = J_eq_2_G(T, G, p, delta, n, alpha_1, alpha_2)

    # Create a larger matrix to hold the combined result
    Jac = np.zeros((2 * (n+1)*(n+1), 2 * (n+1)*(n+1)))
    
    # Insert matrices into appropriate positions
    Jac[:((n+1)*(n+1)), :((n+1)*(n+1))] = Jac_eq_1_T  # Left upper part
    Jac[:((n+1)*(n+1)), ((n+1)*(n+1)):] = Jac_eq_1_G  # Right upper part
    Jac[((n+1)*(n+1)):, :((n+1)*(n+1))] = Jac_eq_2_T  # Left lower part
    Jac[((n+1)*(n+1)):, ((n+1)*(n+1)):] = Jac_eq_2_G  # Right lower part
    
    return Jac

def newton_raphson_system2D(f, J, initial_guess, p, delta, n, alpha_1, alpha_2, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        # Split initial guess into T and G
        T, G = np.split(initial_guess, 2)

        # Evaluate the system of equations and the Jacobian at the current point
        f_val = np.array(f(T, G, p, delta, n, alpha_1, alpha_2))
        J_val = np.array(J(T, G, p, delta, n, alpha_1, alpha_2))

        # Solve the linear system to get the update
        delta_T_G = spsolve(J_val, -f_val)
        
        # Update the solution
        initial_guess += delta_T_G
        print(delta_T_G)
        # Check for convergence
        if np.linalg.norm(delta_T_G) < tol:
            print('Converged after', i + 1, 'iterations.')
            return initial_guess  # Return the solution
    
    raise RuntimeError("Newton-Raphson method did not converge within the maximum number of iterations.")

def initial_guess_2D(xy_min, xy_max, n):
    x = np.linspace(xy_min[0], xy_max[0], n+1)
    y = np.linspace(xy_min[1], xy_max[1], n+1)

    variating = np.linspace(xy_min[0], xy_max[0], (n+1)*(n+1))

    initial_guess = np.zeros(2*(n+1)*(n+1))
    for i in range(0,(n+1)*(n+1)):
        initial_guess[i] = 0.909 + 1 * np.cos(variating[i])
        
    for i in range((n+1)*(n+1), 2*(n+1)*(n+1)):
        initial_guess[i] = 0.5 + 1* np.cos(variating[i-(n+1)*(n+1)])
    return initial_guess

def simulation_with_fire(Initialisation1D, Initialisation2D, plot_solution1D, plot_solution_2D, fire, fire_starting_point = [5000,5000], fire_radius = [2000], dimension = 1, dt = 0.01, num_steps = 100):
    #Check dimension:
    if dimension == 1:

        #Initialisation of the parameters
        dimension, L, n, x_min, x_max, delta, mu, Lambda, dt, num_steps = Initialisation1D()
        #Initial guess
        initial_guess = initial_guess_func(x_min, x_max, n)
        plot_solution1D(initial_guess, n, x_min, x_max, moment = "Initial")


        #Solve the system
        solution = newton_raphson_system(f, J, mu, Lambda, L, delta, n, initial_guess)
        #solution = initial_guess
        #Split solution into T and G
        #Plot the results
        plot_solution1D(solution, n, x_min, x_max, moment = "NR_first")
        # Add Fire
        for fire_rad in fire_radius:
            solution_with_fire = fire(dimension, n, delta, solution, fire_starting_point, fire_rad)
            # Split solution into T and G
            # Plot the results
            plot_solution1D(solution_with_fire, n, x_min, x_max, moment = "After Fire")
            T_values, G_values = solve_time_dependent_equation(solution_with_fire, dt, num_steps, mu, Lambda, L, delta, n)
            x = np.linspace(x_min, x_max, n+1)
            # Plot the evolution of T and G over time on the same plot
            def update(frame):
                plt.cla()  # Clear the current axes

                # Plot T and G for the current frame
                plt.plot(x, T_values[frame], label=f'Time Step {frame} (T)', linestyle='-')
                
                # Set y-axis limits to ensure they remain fixed
                plt.ylim(0, 1)

                # Customize plot
                plt.xlabel('x')
                plt.ylabel('Value')
                plt.title('Evolution of T over Time')
                plt.legend()

            
            # Create the figure and animate the plot
            fig = plt.figure(figsize=(10, 6))
            ani = FuncAnimation(fig, update, frames=num_steps+1, interval=100, repeat=False)
            ani.save('1D_solution_time_propagation.gif', fps=30)
            
        pass
    elif dimension == 2:
        dimension, p, L, n, xy_min, xy_max, delta, alpha_1, alpha_2, dt, num_steps = Initialisation2D()
        initial_guess = initial_guess_2D(xy_min, xy_max, n)
        T_initial, G_initial = np.split(initial_guess, 2)
        plot_solution_2D(T_initial, G_initial, n, moment = "Initial")

        solution = newton_raphson_system2D(f_2D, J_2D, initial_guess, p, delta, n, alpha_1, alpha_2)
        T_solution, G_solution = np.split(solution, 2)
        plot_solution_2D(T_solution, G_solution, n, moment = "NR_first")

        for fire_rad in fire_radius:
            solution_with_fire = fire(dimension, n, delta, solution, fire_starting_point, fire_radius)

            # Split solution into T and G
            T_afterfire, G_afterfire = np.split(solution_with_fire, 2)
            # Plot the results
            plot_solution_2D(T_afterfire, G_afterfire, n, moment = "After Fire")
            T_values, G_values = solve_time_dependent_equation2D(T_initial, G_initial, p, delta, n, alpha_1, alpha_2, dt, num_steps)

            # Reshape T_values and G_values back into 2D grids
            T_values_2d = T_values.reshape((num_steps+1, n+1, n+1))
            G_values_2d = G_values.reshape((num_steps+1, n+1, n+1))

            # Plot the evolution of T and G over time on a 2D grid
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            def update(frame):
                ax[0].cla()
                ax[1].cla()

                im1 = ax[0].imshow(T_values_2d[frame], cmap='viridis', origin='lower', extent=[xy_min[0], xy_max[0], xy_min[0], xy_max[1]])
                ax[0].set_title(f'Time Step {frame} (T)')
                ax[0].set_xlabel('x')
                ax[0].set_ylabel('y')

                im2 = ax[1].imshow(G_values_2d[frame], cmap='viridis', origin='lower', extent=[xy_min[0], xy_max[0], xy_min[0], xy_max[1]])
                ax[1].set_title(f'Time Step {frame} (G)')
                ax[1].set_xlabel('x')
                ax[1].set_ylabel('y')
            
            # Create the figure and animate the plot
            #fig = plt.figure(figsize=(10, 6))
            ani = FuncAnimation(fig, update, frames=num_steps+1, interval=100, repeat=False)
            ani.save('2D_solution_time_propagation.gif', fps=30)



    else:
        print("Error: Something went wrong with the dimensions in the simulation_with_fire function.")

dimension, L, n, x_min, x_max, delta, mu, Lambda, dt, num_steps = Initialisation1D()
#dimension, p, L, n, xy_min, xy_max, delta, alpha_1, alpha_2, dt, num_steps = Initialisation2D()

fire_starting_point = [L/2, L/2]
fire_radius = [0.1*L]
print(fire_radius)

simulation_with_fire(Initialisation1D, Initialisation2D, plot_solution1D, plot_solution_2D, fire, fire_starting_point, fire_radius, dimension, dt, num_steps)




