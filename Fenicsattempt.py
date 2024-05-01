import numpy as np
import fipy as fp

# Define the size of the domain
L, H = 10.0, 10.0  # arbitrary lengths for width and heigh# lo this code is dogshitt
nx, ny = 100, 100  # number of grid points in x and y

# Create a 2D mesh

mesh = fp.Grid2D(nx=nx, ny=ny, dx=L/nx, dy=H/ny)
TildeT = fp.CellVariable(name="T", mesh=mesh, hasOld=True)
TildeG = fp.CellVariable(name="G", mesh=mesh, hasOld=True)

gammaT = 0.75#0.7 #0.85
gammaG = 1.6#0.4 #1.6
DT = 3#1
DG = 24#5
muT = 0.85 #0.2 #0.85
muG = 1.7 #0.27 #1.7
KT = 2 #2
KG = 1 #1
sigmaT = 0.5 #0.05
sigmaG = 2.5 #0.5

p1 = muT/gammaT
p2 = sigmaT*KG/gammaT
p3 = gammaG/gammaT
p4 = muG/gammaT
p6 = sigmaG*KT/gammaT
p5 = DG/DT 

# Example: Zero flux (Neumann condition) at all boundaries
TildeT.faceGrad.constrain(0, where=mesh.exteriorFaces)
TildeG.faceGrad.constrain(0, where=mesh.exteriorFaces)

eqTildeT = (fp.TransientTerm(var=TildeT) == 
            TildeT * (1 - TildeT) - 
            TildeT * (p1 + p2 * TildeG) +
            fp.DiffusionTerm(coeff=1, var=TildeT))

eqTildeG = (fp.TransientTerm(var=TildeG) == 
            p3 * TildeG * (1 - TildeG) - 
            TildeG * (p4 + p5 * TildeT) +
            p6 * fp.DiffusionTerm(coeff=1, var=TildeG))

timeStepDuration = 0.1  # time step size
steps = 200  # number of time steps
TildeT.setValue(1.0)  # example initial condition
TildeG.setValue(0.7)  # example initial condition


for step in range(steps):
    TildeT.updateOld()
    TildeG.updateOld()
    eqTildeT.solve(dt=timeStepDuration)
    eqTildeG.solve(dt=timeStepDuration)
import matplotlib.pyplot as plt


print(np.max(TildeT.value))

# Plotting T
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(TildeT.value.reshape((ny, nx)), origin='lower', extent=(0, L, 0, H))
plt.colorbar()
plt.title('T Distribution')

# Plotting G
plt.subplot(1, 2, 2)
plt.imshow(TildeG.value.reshape((ny, nx)), origin='lower', extent=(0, L, 0, H))
plt.colorbar()
plt.title('G Distribution')
plt.show()
