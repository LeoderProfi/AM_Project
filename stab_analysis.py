# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:05:11 2024

@author: barbe
"""
import numpy as np
import matplotlib.pyplot as plt


# gammaT = 0.75#0.7 #0.85
# gammaG = 1.6#0.4 #1.6
# DT = 3#1
# DG = 24#5
# muT = 0.85 #0.2 #0.85
# muG = 1.7 #0.27 #1.7
# KT = 2 #2
# KG = 1 #1
# sigmaT = 0.5 #0.05
# sigmaG = 2.5 #0.5

gammaT = 3.6#0.7 #0.85
gammaG = 2.3#0.4 #1.6
DT = 0.3#1
DG = 2.4#5
muT = 0.3 #0.2 #0.85
muG = 0.15 #0.27 #1.7
KT = 2 #2
KG = 1 #1
sigmaT = 0.05 #0.05
sigmaG = 0.5 #0.5

p1 = muT/gammaT
p2 = sigmaT*KG/gammaT
p3 = gammaG/gammaT
p4 = muG/gammaT
p5 = sigmaG*KT/gammaT
p6 = DG/DT

p2=2
p6=1/2


print(p3/p2 < (p3-p4)/(1-p1))
print(p5> (p3-p4)/(1-p1))
print(p3/p2>p5)
print((p3*(1-p1) - p2*(p3-p4)))
print((p3-p4)/(1-p1))
L = 10000

u1 = np.array([[0],[0]])
u2 = np.array([[0],[(p3-p4)/p3]])
u3 = np.array([[1-p1],[0]])
#u4 = np.array([[(p3*(1-p1) - p2*(p3-p4))/(p3 - p2*p5)],[(p5*(1-p1) - p2*(p3-p4))/(p2-p3)]])
g0_u4 = ((p3-p4)/p5+p1-1)/(-p2 + p3/p5)
u4 = np.array([[(p3*(1-p1) - p2*(p3-p4))/(p3 - p2*p5)],[g0_u4]])

print(u1)
print(u2)
print(u3)
print(u4)
#print(u4new)
# Geq = ((p3-p4)/p5 + p1 - 1)/((-p2 + p3)/p5)
# u4new = np.array([[1-p1-p2*Geq],[Geq]])

ulist = [u1,u2,u3,u4]#,u4new]


N=1000*100
nlist = np.linspace(0,N,N+1)
lambdalist = [[[],[]],[[],[]],[[],[]],[[],[]]]
for i in range(4):
    T0 = ulist[i][0,0]
    G0 = ulist[i][1,0]
    a1 = 1 - 2*T0 - p2*G0 - p1
    a2 = p3*(1 - 2*G0) - p5*T0 - p4
    b1 = -p2*T0
    b2 = -p5*G0
    d = p6
    for n in nlist:
        cn = a1 - (n*np.pi/L)**2
        dn = a2 - d*(n*np.pi/L)**2
        frac = (cn + dn)/2
        lambdan_1 = frac + np.sqrt(frac**2 + b1*b2 - cn*dn)
        if (type(lambdan_1) != 'float' or type(lambdan_2) != 'float'):
            print('Nannetje!')
            break
        lambdan_2 = frac - np.sqrt(frac**2 + b1*b2 - cn*dn)
        lambdalist[i][0].append(lambdan_1.real)
        # if lambdan_1.real == Nan or lambdan_2.real == Nan:
        #     print('Nannetje! in real')
        #     break
        lambdalist[i][1].append(lambdan_2.real)
    
    plt.figure()
    plt.plot(nlist,lambdalist[i][0], label = 'Re(first eigenvalue)')
    plt.plot(nlist,lambdalist[i][1], label = 'Re(second eigenvalue)')
    # plt.plot([min(nlist), max(nlist)],[0,0], color = 'gray')
    plt.legend()
    plt.title('u_' + str(i+1))
    plt.show()
print(lambdalist[3])

N=1000
nlist = np.linspace(0,N,N+1)
lambdalist = [[[],[]],[[],[]],[[],[]],[[],[]]]

for i in range(4):
    T0 = ulist[i][0,0]
    G0 = ulist[i][1,0]
    a1 = 1 - 2*T0 - p2*G0 - p1
    a2 = p3*(1 - 2*G0) - p5*T0 - p4
    b1 = -p2*T0
    b2 = -p5*G0
    d = p6
    frac = (a1 + a2)/2
    lambdan_1 = frac + np.sqrt(frac**2 + b1*b2 - a1*a2)
    lambdan_2 = frac - np.sqrt(frac**2 + b1*b2 - a1*a2)

    print(lambdan_1)
    print(lambdan_2)
    print()

    # plt.figure()
    # plt.plot(nlist,lambdalist[i][0], label = 'Re(first eigenvalue)')
    # plt.plot(nlist,lambdalist[i][1], label = 'Re(second eigenvalue)')
    # plt.plot([min(nlist), max(nlist)],[0,0], color = 'gray')
    # plt.legend()
    # plt.title('u_' + str(i+1))
    # plt.show()
