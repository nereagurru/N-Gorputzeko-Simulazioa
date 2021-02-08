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
global m, G, N, eps, x, y, vx, vy, aurrekoa, loop, Ebb
N = 2
loop = 1000
G = 8.888*math.pow(10, -10)
eps = math.pow(10, -5)

aurrekoa = np.zeros([N, N])
m = np.array([333000, 1.0])
x = np.array([0.00, 0.983])
y = np.array([0.0, 0.0])
vx = np.array([0.0, 0.0])
vy = np.array([0.0, 0.0175])

Ebb = np.empty((loop))


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

def energia(ax, ay):
    """
        Sistemaren aldiuneko energia kalkulatzeko funtzioa.
        Energia zinetikoa eta potentziala itzultzen ditu.
    """
    global vx, vy, x, y, m
    vxn = vx + ax*h/2
    vyn = vy + ay*h/2
    xij = np.ones([N, N])*x - np.transpose(np.ones([N, N])*x)
    yij = np.ones([N, N])*y - np.transpose(np.ones([N, N])*y)
    R = np.power(np.power(xij, 2) + np.power(yij, 2), 0.5)
    Rlagun = R + np.identity(N)
    matrix = -G*np.transpose(m*np.ones([N, N]))*m/Rlagun
    matrix = matrix - matrix*np.identity(N)
    Ep = np.sum(np.triu(matrix))
    Ez = np.sum(m*(np.power(vxn, 2) + np.power(vyn, 2))/2.0)
    return Ez, Ep

if __name__ == "__main__":
    
    iraupena = time.time()
    h = 1
    aurrekoa_bete()
    ax, ay = axy()
    vx = vx - ax*h/2
    vy = vy - ay*h/2

    fig = plt.figure()
    ardatz = fig.add_subplot(111)

    plt.ylabel(r'$E \; (M_{\oplus} \;AU^{-2} \;egun^{-2})$')
    plt.xlabel('t (egun)')

    fig.text(0.25, 0.95, "POTENTZIALA", ha="center", va="bottom", size="large",color="purple")
    fig.text(0.5, 0.95, "TOTALA", ha="center", va="bottom", size="large")
    fig.text(0.75,0.95,"ZINETIKOA", ha="center", va="bottom", size="large",color="green")
    plt.ylim((-0.0004, 0.0002))

    for t in range(0, loop):
        ax, ay = axy()
        vx = vx + ax*h
        vy = vy + ay*h
        x = x + vx*h
        y = y + vy*h
        ez, ep = energia(ax, ay)
        if t%(loop/50) == 0:    
            plt.scatter(t*h, ez, color = 'purple')
            plt.scatter(t*h, ep, color = 'green')
            plt.scatter(t*h, ez + ep, color = 'black')
        Ebb[t] = ez + ep

print('Sistema honen bataz besteko energia ' + str(round(np.average(Ebb), 6)) + ' da eta desbiderapen estandarra ' + str(round(np.std(Ebb), 9)))
print('Programa honek ' + str(round(time.time() - iraupena, 2)) + 's behar izan ditu exekutatzeko')         

plt.savefig(fname='EnergiaTotala.png')
plt.show()
