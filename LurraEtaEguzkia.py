# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:29:23 2021

@author: nerea
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

# --------KONSTANTEAK-------------#
global m, G, N, eps, x, y, vx, vy
N = 2
G = 8.888*math.pow(10, -10)
eps = math.pow(10, -5)


m = np.array([333000, 1.0])
x = np.array([0.00, 0.983])
y = np.array([0.0, 0.0])
vx = np.array([0.0, 0.0])
vy = np.array([0.0, 0.0175])


def axy():
    global x, y, eps, N, m
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

    return np.sum(matrixx, axis=1), np.sum(matrixy, axis=1)


if __name__ == "__main__":
    
    aber = time.time()
    h = 1
    ax, ay = axy()
    vx = vx - ax*h/2
    vy = vy - ay*h/2

    fig = plt.figure()
    ardatz = fig.add_subplot(111)
    plt.scatter(x[0], y[0], label='Eguzkia' , color='blue', linewidths=0.5)
    plt.scatter(x[1], y[1], label='Lurra' , color='red', linewidths=0.5)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.legend()
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))

    ardatz.set_aspect('equal', adjustable='box')
    kont = 0
    for t in range(0, 400):
        ax, ay = axy()
        vx = vx + ax*h
        vy = vy + ay*h
        x = x + vx*h
        y = y + vy*h
        if t%10 == 0:    
            plt.scatter(x[0], y[0], color = 'blue', linewidths=0.5)
            plt.scatter(x[1], y[1], color = 'red', linewidths=0.5)
            
        if t%100 == 0 or t%365 == 0:
            izena = str(kont) + '.png'
            plt.title('t = ' + str(h*t) + ' egun\n')
            plt.savefig(fname = izena)
            kont = kont + 1

print('Programa honek ' + str(aber - time.time()) + 's behar izan ditu exekutatzeko')
