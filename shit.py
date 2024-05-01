import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt




gammaT = 3.6
gammaG = 2.3
DT = 0.3
DG = 2.4
muT = 0.3
muG = 0.15
KT = 2.0
KG = 1
sigmaT = 0.05
sigmaG = 0.5

p_1 = muT/gammaT
p_2 = sigmaT*KG/gammaT
p_3 = gammaG/gammaT
p_4 = muG/gammaT
p_6 = sigmaG*KT/gammaT
p_5 = DG/DT


L = 10000
n = 40
x_min = 0
x_max = L
y_min = 0
y_max = L
delta_x = (x_max - x_min) / n


alpha_1 = DT/(gammaT*L**2)
alpha_2 = DG/(gammaT*L**2)





def f(T, G):
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

def J_eq_1_T(T, G):
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

def J_eq_1_G(T, G):
    # Generate the Jacobian matrix
    jac_matrix = np.zeros(((n+1)*(n+1), (n+1)*(n+1)))
    for i in range(0, (n+1)*(n+1)):
        jac_matrix[i, i] = -p_2 * T[i]
    return jac_matrix

def J_eq_2_T(T, G):
    # Generate the Jacobian matrix
    jac_matrix = np.zeros(((n+1)*(n+1), (n+1)*(n+1)))
    for i in range(0, (n+1)*(n+1)):
        jac_matrix[i, i] = -p_6 * G[i]
    return jac_matrix

def J_eq_2_G(T, G):
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

def J(T, G):
    Jac_eq_1_T = J_eq_1_T(T, G)
    Jac_eq_1_G = J_eq_1_G(T, G)
    Jac_eq_2_T = J_eq_2_T(T, G)
    Jac_eq_2_G = J_eq_2_G(T, G)

    # Create a larger matrix to hold the combined result
    Jac = np.zeros((2 * (n+1)*(n+1), 2 * (n+1)*(n+1)))
   
    # Insert matrices into appropriate positions
    Jac[:((n+1)*(n+1)), :((n+1)*(n+1))] = Jac_eq_1_T  # Left upper part
    Jac[:((n+1)*(n+1)), ((n+1)*(n+1)):] = Jac_eq_1_G  # Right upper part
    Jac[((n+1)*(n+1)):, :((n+1)*(n+1))] = Jac_eq_2_T  # Left lower part
    Jac[((n+1)*(n+1)):, ((n+1)*(n+1)):] = Jac_eq_2_G  # Right lower part
   
    return Jac



def newton_raphson_system(f, J, initial_guess, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        # Split initial guess into T and G
        T, G = np.split(initial_guess, 2)

        # Evaluate the system of equations and the Jacobian at the current point
        f_val = np.array(f(T, G))
        J_val = np.array(J(T, G))

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




# Initial guess
x = np.linspace(x_min, x_max, n+1)
y = np.linspace(y_min, y_max, n+1)

variating = np.linspace(x_min, x_max, (n+1)*(n+1))

initial_guess = np.zeros(2*(n+1)*(n+1))
for i in range(0,(n+1)*(n+1)):
    initial_guess[i] = 0.909+ 1 * np.cos(variating[i])
   
for i in range((n+1)*(n+1), 2*(n+1)*(n+1)):
    initial_guess[i] = 0.5 + 1* np.cos(variating[i-(n+1)*(n+1)])
   




# Solve the system
solution = newton_raphson_system(f, J, initial_guess)

# Split solution into T and G
T_solution, G_solution = np.split(solution, 2)


Jac = J(T_solution, G_solution)

eigenvalues = np.linalg.eigvals(Jac)

x_vertices = (n+1)
y_vertices = (n+1)

T_reshaped = np.reshape(T_solution, (x_vertices,y_vertices))
G_reshaped = np.reshape(G_solution, (x_vertices,y_vertices))





# Plotting both images together
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a grid of 1 row and 2 columns

# Plot the first image (T)
im1 = axs[0].imshow(T_reshaped, extent=[0, 1, 1, 0])
axs[0].set_title('T')  # Set title for the subplot
fig.colorbar(im1, ax=axs[0])  # Add colorbar to the subplot
axs[0].set_ylabel('y-axis')
axs[0].set_xlabel('x-axis')

# Plot the second image (G)
im2 = axs[1].imshow(G_reshaped, extent=[0, 1, 1, 0])
axs[1].set_title('G')  # Set title for the subplot
fig.colorbar(im2, ax=axs[1])  # Add colorbar to the subplot
axs[1].set_ylabel('y-axis')
axs[1].set_xlabel('x-axis')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


