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

        X = np.arange(n_cells) #Create an array of the x-values 

        distance = np.abs(X - fire_starting_index_x) #Calculate the distance from the starting point in which the fire should be active 

        mask = distance <= fire_radius/delta #Create a mask for the cells that should be affected by the fire 

        ST[mask] = 0.2 * ST[mask] #Set the Tree concentration of the cells affected by the fire to 0 

        #SG[mask] = 0.0 #Set the gras concentration of the cells affected by the fire to 0 

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

# New model 1D 

def Initialisation1D_new(): 

    L =100; n = 4000; x_min = 0; x_max = L; delta_x = (x_max - x_min) / n 
    A = 2.2; B = 0.45; nu = 182.5  

    # Time parameters 
    dt = 1.1*(10**-4) # Time step size 
    num_steps = 100000  # Number of time steps we need these many because its a slow process :(

    p = { 
        'A': A, 
        'B': B, 
        'nu': nu, 
        'n_cells': n, 
        'x_min': x_min, 
        'x_max': x_max, 
        'delta_x': delta_x, 
        'time_step_size': dt, 
        'num_time_steps': num_steps 
    } 
    return p 


def rk4_newModel_step(u_n, w_n, dt): 

    """ 
    Perform one step of the fourth-order Runge-Kutta method. 
    Parameters: 
    - u_n, w_n: Current values of u and w at time step n. 
    - dt: Time step size. 
    - f: Function representing the right-hand side of the equation. 
    Returns: 
    - u_np1, w_np1: Values of u at the next time step (n+1). 
    """ 

    p = Initialisation1D_new() 
    A = p['A'] 
    B = p['B'] 
    nu = p['nu'] 
    n = p['n_cells'] 
    delta_x = p['delta_x'] 

    def f_u(u, w): 

        equations_u = [w[i]*u[i]**2 - B*u[i] + (u[i+1] - 2 * u[i] + u[i-1]) / delta_x**2 for i in range(1, n-1)] 
        equations_u.insert(0,w[0]*u[0]**2 - B*u[0] + (u[1] - 2 * u[0] + u[n-1]) / delta_x**2) 
        equations_u.append(w[n-1]*u[n-1]**2 - B*u[n-1] + (u[0] - 2 * u[n-1] + u[n-2]) / delta_x**2) 
        return np.array(equations_u) 

    def f_w(u, w): 

        equations_w = [A - w[i] - w[i]*u[i]**2 + nu*(w[i+1] - w[i]) / (2*delta_x) for i in range(1, n-1)] 
        equations_w.insert(0, A - w[0] - w[0]*u[0]**2 + nu*(w[1] - w[n-1])/(2*delta_x)) 
        equations_w.append(A - w[n-1] - w[n-1]*u[n-1]**2 + nu*(w[0] - w[n-2])/(2*delta_x)) 

        return np.array(equations_w) 

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

 
def solve_time_dependent_equation(u_initial, w_initial, dt, num_steps, res = 1e-3): 
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
    residual = 1 
    counter = 0 

    #for _ in range(num_steps): 

    while residual > res and counter < num_steps: 

        u_np1, w_np1 = rk4_newModel_step(u_n, w_n, dt) 
        u_values.append(u_np1) 
        w_values.append(w_np1) 
        residual = np.linalg.norm(u_np1 - u_n) #+ np.linalg.norm(w_np1 - w_n) 
        u_n = u_np1.copy() 
        w_n = w_np1.copy() 

        if counter % 1000 == 0: 
            print(counter, residual)               
        counter += 1 

    return np.array(u_values), np.array(w_values) 

 
def initial_guess_newModel(x_min, x_max, n): 

    x = np.linspace(x_min, x_max, n) 

    L = x_max - x_min 

    initial_guess = np.zeros(2*(n)) 

    for i in range(0,n): 

        initial_guess[i] = (7.5+7.5*np.sin(10 * np.pi * x[i] / L + 2))#random.uniform(0.95*stable_eq_u, 1.05*stable_eq_u)   

    for i in range(n, 2*(n)): 

        initial_guess[i] = (0.1+0.1*np.sin(10 * np.pi  * x[i-n]/ L)) #random.uniform(0.95*stable_eq_w, 1.05*stable_eq_w) 

    u_initial, w_initial = np.split(initial_guess, 2) 

    return u_initial, w_initial 

def fire_new_model(): 
    results = {}
    p = Initialisation1D_new() 
    n = p['n_cells'] 
    
    x_min = p['x_min'] 
    x_max = p['x_max'] 

    delta_x = p['delta_x'] 
    dt = p['time_step_size'] 

    num_steps = p['num_time_steps'] 
    u_initial, w_initial = initial_guess_newModel(x_min, x_max, n) 
    u_values, w_values = solve_time_dependent_equation(u_initial, w_initial, dt, num_steps, res = 1e-3) 
    solution = np.concatenate((u_values[-1], w_values[-1])) 

    np.save('Initial_u_w', solution, allow_pickle=True, fix_imports=True)

    u_values, w_values = np.split(solution, 2)

    results['Initial_u'] = u_values
    results['Initial_w'] = u_values

    dimension = 1 

    fire_starting_point = [(x_max-x_min)/2, (x_max-x_min)/2] 
    fire_radius = [0.15*(x_max-x_min), 0.15*(x_max-x_min)] 

    counter = 0
    for fire_rad in fire_radius: 
        counter +=1
 
        solution = fire(dimension, n, delta_x, solution, fire_starting_point, fire_rad) 
        np.save('AfterFire_u_w_{}'.format(counter), solution, allow_pickle=True, fix_imports=True)

        T_afterfire, G_afterfire = np.split(solution, 2) 

        results['AfterFire_u_{}'.format(counter)] = T_afterfire
        results['AfterFire_w_{}'.format(counter)] = G_afterfire

        # Plot the results 

        u_values, w_values = solve_time_dependent_equation(T_afterfire, G_afterfire, dt, num_steps, res = 1e-3)

        results['AfterFire_u_converged_{}'.format(counter)] = u_values
        results['AfterFire_w_converged_{}'.format(counter)] = w_values

        solution = np.concatenate((u_values[-1], w_values[-1]))

        np.save('AfterFire_u_w_converged_{}'.format(counter), solution, allow_pickle=True, fix_imports=True)

    return results

results = fire_new_model()

plt.plot(results['Initial_u'].flatten(), label = 'Veg. Initial')
plt.plot(results['AfterFire_u_1'].flatten(), label = 'Veg. After first fire')
plt.plot(results['AfterFire_u_converged_1'][-1].flatten(), label = 'Veg. After first fire converged')
plt.legend(loc='upper right')
plt.savefig('fire_new_model_1.pdf')
plt.clf()
plt.plot(results['AfterFire_u_converged_1'][-1].flatten(), label = 'Veg. After first fire converged')
plt.plot(results['AfterFire_u_2'].flatten(), label = 'Veg. After second fire')
plt.plot(results['AfterFire_u_converged_2'][-1].flatten(), label = 'Veg. After second fire converged')
plt.legend(loc='upper right')
plt.savefig('fire_new_model_2.pdf')
plt.clf()


total_biomass = []
total_biomass.append(np.sum(results['Initial_u']))
total_biomass.append(np.sum(results['AfterFire_u_1']))
total_biomass.append(np.sum(results['AfterFire_u_converged_1'][-1]))
total_biomass.append(np.sum(results['AfterFire_u_2']))
total_biomass.append(np.sum(results['AfterFire_u_converged_2'][-1]))
x = [0,1,2,3,4]


plt.plot(x, total_biomass, label = 'Total biomass')
plt.legend()
plt.savefig('fire_new_model_total_biomass.pdf')
plt.clf()


#plt.plot(results['Initial_w'].flatten(), label = 'Initial w')
plt.plot(results['AfterFire_w_1'].flatten(), label = 'Water after first fire')
plt.plot(results['AfterFire_w_converged_1'][-1].flatten(), label = 'Water After first fire converged')
#plt.plot(results['AfterFire_w_2'].flatten(), label = 'Water after secoond fire')
plt.plot(results['AfterFire_w_converged_2'][-1].flatten(), label = 'Water after second fire converged')
plt.legend(loc='upper right')
plt.savefig('fire_new_model_w.pdf')