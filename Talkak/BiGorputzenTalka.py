# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:02:04 2021

@author: nerea
"""
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


# --------KONSTANTEAK-------------#
global m, k, N, eps, x, y, z, vxn, vyn, vyz, loop
N = 2
D = 0.05
loop = 10000
k = 10
mtot = 1.0
G = 1
w = math.pow(k/mtot, 0.5)
gamma = w
"""
gamma < w jarriz gero bi partikulak oszilatzen hasi itsastea lortzen bagute. v handiegia bada ez dute lortuko alde egitea.
gamma > w jarriz gero bi partikulak itsatsi eta azkenean energia osoa galdu
KONTUZ! abiadura handiegia bada w gamma nahiko handia izan behar da partikulak itsasteko, baina ez handiegia gehiegi alderanzteko.
"""
m = np.full((N), mtot)
x = np.array([1.0, 2.0])
y = np.array([1.0, 1.0])
z = np.array([1.0, 1.0])
vxn = np.array([1.0, -1.0])
vyn = np.full((N), 0.0)
vzn = np.full((N), 0.0)

koont = 0
def axyz():
    global x, y, z, vxn, vyn, vzn, eps, N, lagun, x
    """
    Grabitatearen ondorioz elkarrekintzak jasatzen dituzten partikulen
    azelerazioa kalkulatzeko funtzioa 2D-tan.

    ax, ay itzuli
    """
    xij = np.ones([N, N])*x - np.transpose(np.ones([N, N])*x)
    yij = np.ones([N, N])*y - np.transpose(np.ones([N, N])*y)
    zij = np.ones([N, N])*z - np.transpose(np.ones([N, N])*z)
    vxij = np.ones([N, N])*vxn - np.transpose(np.ones([N, N])*vxn)
    vyij = np.ones([N, N])*vyn - np.transpose(np.ones([N, N])*vyn)
    vzij = np.ones([N, N])*vyn - np.transpose(np.ones([N, N])*vyn)
    R = np.power(np.power(xij, 2) + np.power(yij, 2), 0.5)  # R
    grab = np.logical_and(R>D, True)
    matrixx = np.zeros((N, N))
    matrixy = np.zeros((N, N))
    matrixz = np.zeros((N, N))
    Rlagun = R + np.identity(N)
    matrixx = grab*(G*m*np.power(Rlagun, -3))*xij + np.logical_not(grab)*(w*w*xij + 2*gamma*vxij)/np.transpose(m)
    matrixy = grab*(G*m*np.power(Rlagun, -3))*yij + np.logical_not(grab)*(w*w*yij + 2*gamma*vyij)/np.transpose(m)
    matrixz = grab*(G*m*np.power(Rlagun, -3))*zij + np.logical_not(grab)*(w*w*zij + 2*gamma*vzij)/np.transpose(m)

    return np.sum(matrixx, axis=1), np.sum(matrixy, axis=1), np.sum(matrixy, axis=1)


if __name__ == "__main__":

    iraupena = time.time()
    h = 0.001

    fig = plt.figure()

    axx = fig.gca(projection='3d')
    axx.set_xlim3d(0, 2)
    axx.set_ylim3d(0, 2)
    axx.set_zlim3d(0, 2)
    
    ax, ay, az = axyz()
    vx = vxn - ax*h/2
    vy = vyn - ay*h/2
    vz = vzn - az*h/2
    kont = 0
    # ardatz.set_aspect('equal', adjustable='box')
    for t in range(0, loop):
        ax, ay, az = axyz()
        vx = vx + ax*h
        vy = vy + ay*h
        vz = vz + az*h
        x = x + vx*h
        y = y + vy*h
        z = z + vz*h
        vxn = vx - ax*h/2
        vyn = vy - ay*h/2
        vzn = vz - az*h/2
        if t % 500 == 0:
            axx = fig.gca(projection='3d')
            axx.set_xlim3d(0, 2)
            axx.set_ylim3d(0, 2)
            axx.set_zlim3d(0, 2)
            for i in range(0, N2):
                plt.scatter(x[i], y[i], z[i], color='blue', linewidths=10)
                plt.scatter(x[i + N2], y[i + N2], z[i + N2], color='red', linewidths=10)
            izena = str(kont) + '.jpg'
            plt.title('t = ' + str(h*t) + ' segundu\n')
            plt.savefig(fname = izena)
            kont = kont + 1
            plt.clf()

print('Programa honek ' + str(round(time.time() - iraupena, 2)) +
      's behar izan ditu exekutatzeko.')

plt.show()
