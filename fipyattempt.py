import numpy as np
import matplotlib.pyplot as plt
import fipy as fp
from fipy import CellVariable, Grid2D, Viewer, TransientTerm, DiffusionTerm
from fipy import PeriodicGrid2D, CellVariable, Viewer
from fipy.tools import numerix

# Create a 2D grid with periodic boundary conditions
mesh = PeriodicGrid2D(nx=100, ny=100, dx=0.01, dy=0.01)

# Create cell variables for TildeT and TildeG
TildeT = CellVariable(name="TildeT", mesh=mesh)
TildeG = CellVariable(name="TildeG", mesh=mesh)


# Define the size of the domain
L, H = 1.0, 1.0  # arbitrary lengths for width and height
nx, ny = 40, 40  # number of grid points in x and y

# Create a 2D mesh

#mesh = fp.Grid2D(nx=nx, ny=ny, dx=L/nx, dy=H/ny)
mesh = PeriodicGrid2D(nx=nx, ny=ny, dx=L/nx, dy=L/ny)

"""TildeT = fp.CellVariable(name="T", mesh=mesh, hasOld=True)
TildeG = fp.CellVariable(name="G", mesh=mesh, hasOld=True)"""
# Create cell variables for TildeT and TildeG
TildeT = CellVariable(name="TildeT", mesh=mesh)
TildeG = CellVariable(name="TildeG", mesh=mesh)
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
p5 = sigmaG*KT/gammaT
alpha_1 = DT/(gammaT*L**2)
alpha_2 = DG/(gammaT*L**2)

# Example: Zero flux (Neumann condition) at all boundaries
TildeT.faceGrad.constrain(0, where=mesh.exteriorFaces)
TildeG.faceGrad.constrain(0, where=mesh.exteriorFaces)

eqTildeT = (fp.TransientTerm(var=TildeT) == 
            TildeT * (1 - TildeT) - 
            TildeT * (p1 + p2 * TildeG) + alpha_1 *
            fp.DiffusionTerm(coeff=1, var=TildeT))

eqTildeG = (fp.TransientTerm(var=TildeG) == 
            p3 * TildeG * (1 - TildeG) - 
            TildeG * (p4 + p5 * TildeT) +
            alpha_2 * fp.DiffusionTerm(coeff=1, var=TildeG))

timeStepDuration = 0.005  # time step size
steps = 10000 # number of time steps

x = np.linspace(0, L, nx*ny)
#TildeT.setValue(np.sin(x)+1)
#TildeG.setValue(np.cos(x)+1)
#TildeT.setValue(np.random.rand(*x.shape))
#TildeG.setValue(np.random.rand(*x.shape))
# Set the initial conditions for TildeT and TildeG
"""TildeT.setValue(numerix.random((mesh.nx, mesh.ny)))
TildeG.setValue(numerix.random((mesh.nx, mesh.ny)))"""
TildeT.setValue(numerix.random.random_sample((mesh.nx, mesh.ny)))
TildeG.setValue(numerix.random.random_sample((mesh.nx, mesh.ny)))

#TildeT.setValue(0.5)  # example initial condition
#TildeG.setValue(0.7)  # example initial condition

# Create viewers for TildeT and TildeG
viewerT = Viewer(vars=TildeT)
viewerG = Viewer(vars=TildeG)

for step in range(steps):
    TildeT.updateOld()
    TildeG.updateOld()
    eqTildeT.solve(dt=timeStepDuration)
    eqTildeG.solve(dt=timeStepDuration)
    
    # Plot TildeT and TildeG separately
    viewerT.plot()
    viewerG.plot()




"""# Plotting T
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
plt.show()"""
