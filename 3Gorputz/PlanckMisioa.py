# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:49:46 2021

@author: nerea
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt


# --------KONSTANTEAK-------------#
global m, G, N, eps, x, y, vx, vy, aurrekoa, loop, Ebb
N = 2
loop = 10000000
G = 8.888*math.pow(10, -10)
eps = math.pow(10, -5)


aurrekoa = np.zeros([N, N])

m = np.array([333000, 1.0])
x = np.array([0.00, 0.983])
y = np.array([0.0, 0.0])
vx = np.array([0.0, 0.0])
vy = np.array([0.0, 0.0175])


def axy():
    global x, y, aurrekoa, eps, N, m
    """
    Grabitatearen ondorioz elkarrekintzak jasatzen dituzten partikulen
    azelerazioa kalkulatzeko funtzioa 2D-tan.

    ax, ay itzuli
    """

    xij = np.ones([N, N])*x - np.transpose(np.ones([N, N])*x)
    yij = np.ones([N, N])*y - np.transpose(np.ones([N, N])*y)
    R = np.power(np.power(xij, 2) + np.power(yij, 2), 0.5)
    matrixx = np.zeros((N, N))
    matrixy = np.zeros((N, N))
    Rlagun = R + np.identity(N)

    matrixx = (G*m*np.power(Rlagun, -3))*xij
    matrixy = (G*m*np.power(Rlagun, -3))*yij
    aurrekoa = Rlagun - np.identity(N)*Rlagun

    return np.sum(matrixx, axis=1), np.sum(matrixy, axis=1)


def aurrekoa_bete():
    global aurrekoa, x, y, N
    """
    aurrekoa aldagai globala hasieratu partikulen arteko distantziekin
    """
    xij = np.power(np.ones([N, N])*x - np.transpose(np.ones([N, N])*x), 2)
    yij = np.power(np.ones([N, N])*y - np.transpose(np.ones([N, N])*y), 2)
    aurrekoa = np.power(xij + yij, 0.5)


if __name__ == "__main__":

    iraupena = time.time()
    h = 0.0001
    aurrekoa_bete()
    ax, ay = axy()
    vx = vx - ax*h/2
    vy = vy - ay*h/2
    fig = plt.figure()
    ardatz = fig.add_subplot(111)

    plt.ylabel('y (AU)')
    plt.xlabel('x (AU)')
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    kont = 0
    ardatz.set_aspect('equal', adjustable='box')
    for t in range(0, loop):
        ax, ay = axy()
        vx = vx + ax*h
        vy = vy + ay*h
        x = x + vx*h
        y = y + vy*h
        if h*t == 184:
            plt.scatter(x[0], y[0], color='blue')
            plt.scatter(x[1], y[1], color='red')
            plt.savefig(fname=(str(kont) + '.png'))
            kont = kont + 1
            break

    N = 3
    angelua = math.atan(y[1]/x[1]) + math.pi  # 2.koadrantea
    m = np.array([333000, 1.0, 3.1808*math.pow(10, -22)])
    x = np.array([x[0], x[1], 1.004*math.cos(angelua)])
    y = np.array([y[0], y[1], 1.004*math.sin(angelua)])
    vx = np.array([vx[0], vx[1], vx[1]*x[1]/x[2]])
    vy = np.array([vy[0], vy[1], vy[1]*y[1]/y[2]])

    print('Satelitearen hasierako informazioa:\n')
    print('x = ' + str(round(x[2], 3)) + ' AU')
    print('y = ' + str(round(y[2], 3)) + ' AU')
    print('vx = ' + str(round(vx[2], 7)) + ' AU/egun')
    print('vy = ' + str(round(vy[2], 7)) + ' AU/egun')

    for t in range(0, loop):
        ax, ay = axy()
        vx = vx + ax*h
        vy = vy + ay*h
        x = x + vx*h
        y = y + vy*h
        if (h*t) % 10 == 0:
            plt.scatter(x[0], y[0], color='blue')
            plt.scatter(x[1], y[1], color='red')
            plt.scatter(x[2], y[2], color='black')
            plt.title('t = ' + str(h*t) + ' egun')
            plt.savefig(fname=(str(kont) + '.png'))
            kont = kont + 1
        if h*t == 365:
            plt.scatter(x[0], y[0], color='blue')
            plt.scatter(x[1], y[1], color='red')
            plt.scatter(x[2], y[2], color='black')
            plt.title('t = ' + str(h*t) + 'egun')
            plt.savefig(fname=(str(kont) + '.png'))
            break

print('Programa honek ' + str(round(time.time() - iraupena, 2)) +
      's behar izan ditu exekutatzeko.')
plt.show()
