import numpy as np
import math 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

N = 100
V1 = 0
V2 = 20
V3 = 100
potential = ti.field(dtype=ti.f64, shape = (N,N))
E = ti.Vector.field(n=2, shape = (N, N), dtype=ti.f64)
@ti.kernel
def boundary_condition():
    for i, j in potential:
        if j == 0:
            potential[i,j] = V1
        if j == N-1:
            potential[i, j] = V3
            
        if i == 0 or i == N-1:
            if 0.45* N < j < (1-0.45 )*N:
                potential[i, j] = V2


boundary_condition()


@ti.kernel
def update():
    for i, j in potential:
        if j ==0 or j == N-1:
            potential[i, j] = potential[i, j]
        else:
            if i == 0:
                a = potential[i, j] + potential[i + 1, j] + potential[i, j+1] + potential[i, j-1]
                potential[i, j] = 0.25 * a
            elif i == N-1 :
                a =  potential[i, j] + potential[i - 1, j] + potential[i, j+1] + potential[i, j-1]
                potential[i, j] = 0.25 * a
            else:
                a = potential[i+1, j] + potential[i - 1, j] + potential[i, j+1] + potential[i, j-1]
                potential[i, j] = 0.25 * a




for i in range(30000):
    update()
    boundary_condition()


potential_array = potential.to_numpy()
plt.matshow(potential_array)
plt.colorbar()


@ti.kernel
def potential_to_E():
    for k, l in potential:
        if k != 0 and k != N-1:
            if l != 0 and l != N-1:
                x =- (potential[k+1, l] - potential[k-1, l])/2
                y =- (potential[k, l+1] - potential[k, l-1])/2
                E[k,l] = ti.math.vec2(x, y)


potential_to_E()

X, Y = np.mgrid[0:N, 0:N]
U = np.zeros((N, N))
V = np.zeros((N, N))
E_list = E.to_numpy()


for i in range(N):
    for j in range(N):
        U[i][j] = E_list[i][j][0]
        V[i][j] = E_list[i][j][1]

fig1 = plt.figure(1, figsize=(20,20))
plt.streamplot(X, Y, V, U, density=0.5)