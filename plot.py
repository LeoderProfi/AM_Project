import numpy as np
import matplotlib.pyplot as plt

# Data for plotting
X1 = np.load("Initial_u_w.npy")
X2 =np.load("AfterFire_u_w_1.npy")
X3 =np.load("AfterFire_u_w_converged_1.npy")
X4 =np.load("AfterFire_u_w_2.npy")
X5 =np.load("AfterFire_u_w_converged_2.npy")


u1, w1 =np.split(X1, 2)
u2, w2 =np.split(X2, 2)
u3, w3 =np.split(X3, 2)
u4, w4 =np.split(X4, 2)
u5, w5 =np.split(X5, 2)

plt.plot(u1, label='Veg. Initial', color='black')
plt.plot(u2, label='Veg. After Fire 1', color='red')
plt.plot(u3, label='Veg. After Fire 1 Converged', color='blue')
plt.legend()
plt.savefig('veg_fire1.pdf')
plt.clf()


plt.plot(u3, label='Veg. After Fire 1 Converged', color='blue')
plt.plot(u4, label='Veg. After Fire 2', color='green')
plt.plot(u5, label='Veg. After Fire 2 Converged', color='purple')
plt.legend()
plt.savefig('veg_fire2.pdf')
plt.clf()

"""biomass = [np.sum(u1), np.sum(u2), np.sum(u3), np.sum(u4), np.sum(u5)]
plt.xticks(np.arange(5), ['Initial', 'Fire 1', 'Fire 1 Conv,', 'Fire 2', 'Fire 2 Conv.'])
plt.plot(biomass, label='Biomass', linestyle='--', marker='o')
plt.show()"""

#plt.plot(w1)
plt.plot(w2, label='Water After Fire 1')
plt.plot(w3, label='Water After Fire 1 Converged')
#plt.plot(w4)
plt.plot(w5, label='Water After Fire 2 Converged')
plt.legend()
plt.savefig('water_fire.pdf')