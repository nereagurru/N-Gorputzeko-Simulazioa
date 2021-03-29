# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:23:06 2021

@author: nerea
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from skyfield.api import load
import sys


ts = load.timescale()
t = ts.now()

planets = load('de421.bsp')

sun = planets['sun']
earth = planets['earth']
mars = planets['mars']
mercury = planets['mercury']
venus = planets['venus']
jupiter = planets['JUPITER BARYCENTER']
saturn = planets['SATURN BARYCENTER']
uranus = planets['URANUS BARYCENTER']
neptune = planets['NEPTUNE BARYCENTER']
pluton = planets['PLUTO BARYCENTER']
sys.exit
sun_pos = sun.at(t).position.au
earth_pos = earth.at(t).position.au
mars_pos = mars.at(t).position.au
mercury_pos = mercury.at(t).position.au
venus_pos = venus.at(t).position.au
jupiter_pos = jupiter.at(t).position.au
saturn_pos = saturn.at(t).position.au
uranus_pos = uranus.at(t).position.au
neptune_pos = neptune.at(t).position.au
pluton_pos = pluton.at(t).position.au

sun_vel = sun.at(t).velocity.au_per_d
earth_vel = earth.at(t).velocity.au_per_d
mars_vel = mars.at(t).velocity.au_per_d
mercury_vel = mercury.at(t).velocity.au_per_d
venus_vel = venus.at(t).velocity.au_per_d
jupiter_vel = jupiter.at(t).velocity.au_per_d
saturn_vel = saturn.at(t).velocity.au_per_d
uranus_vel = uranus.at(t).velocity.au_per_d
neptune_vel = neptune.at(t).velocity.au_per_d
pluton_vel = pluton.at(t).velocity.au_per_d


global m, G, N, eps, x, y, z, vx, vy, vz
m = np.array([333000, 0.05528, 0.81500, 1.0, 0.10745,
              317.83, 95.159, 14.536, 17.147, 0.0021])
N = 10
G = 8.888*math.pow(10, -10)
eps = math.pow(10, -2)


x = np.array([sun_pos[0], mercury_pos[0], venus_pos[0], earth_pos[0],
              mars_pos[0], jupiter_pos[0], saturn_pos[0], uranus_pos[0],
              neptune_pos[0], pluton_pos[0]])
y = np.array([sun_pos[1], mercury_pos[1], venus_pos[1], earth_pos[1],
              mars_pos[1], jupiter_pos[1], saturn_pos[1], uranus_pos[1],
              neptune_pos[1], pluton_pos[1]])
z = np.array([sun_pos[2], mercury_pos[2], venus_pos[2], earth_pos[2],
              mars_pos[2], jupiter_pos[2], saturn_pos[2], uranus_pos[2],
              neptune_pos[2], pluton_pos[2]])

hasierax = x
hasieray = y
hasieraz = z

vx = np.array([sun_vel[0],  mercury_vel[0], venus_vel[0], earth_vel[0],
               mars_vel[0], jupiter_vel[0], saturn_vel[0], uranus_vel[0],
               neptune_vel[0], pluton_vel[0]])
vy = np.array([sun_vel[1],  mercury_vel[1], venus_vel[1], earth_vel[1],
               mars_vel[1], jupiter_vel[1], saturn_vel[1], uranus_vel[1],
               neptune_vel[1], pluton_vel[1]])
vz = np.array([sun_vel[2],  mercury_vel[2], venus_vel[2], earth_vel[2],
               mars_vel[2], jupiter_vel[2], saturn_vel[2], uranus_vel[2],
               neptune_vel[2], pluton_vel[2]])


def axyz():
    global x, y, z, eps, N, m
    """
    Grabitatearen ondorioz elkarrekintzak jasatzen dituzten partikulen
    azelerazioa kalkulatzeko funtzioa 3D-tan.

    ax, ay, az itzuli
    """

    xij = np.ones([N, N])*x - np.transpose(np.ones([N, N])*x)
    yij = np.ones([N, N])*y - np.transpose(np.ones([N, N])*y)
    zij = np.ones([N, N])*z - np.transpose(np.ones([N, N])*z)
    R = np.power(np.power(xij, 2) + np.power(yij, 2) + np.power(zij, 2), 0.5)

    matrixx = np.zeros((N, N))
    matrixy = np.zeros((N, N))
    matrixz = np.zeros((N, N))
    Rlagun = R + np.identity(N)

    matrixx = (G*m*np.power(Rlagun, -3))*xij
    matrixy = (G*m*np.power(Rlagun, -3))*yij
    matrixz = (G*m*np.power(Rlagun, -3))*zij

    ax = np.sum(matrixx, axis=1)
    ay = np.sum(matrixy, axis=1)
    az = np.sum(matrixz, axis=1)
    return ax, ay, az


if __name__ == "__main__":
    exet = time.time()
    h = 1

    ax, ay, az = axyz()
    vx = vx - ax*h/2
    vy = vy - ay*h/2
    vz = vz - az*h/2

    per = np.full([N, 1], None)
    for t in range(1, 100000):
        ax, ay, az = axyz()
        vx = vx + ax*h
        vy = vy + ay*h
        vz = vz + az*h
        x = x + vx*h
        y = y + vy*h
        z = z + vz*h
        per_bete = np.logical_and(abs(x - hasierax) < eps,
                                  abs(y - hasieray) < eps,
                                  abs(z - hasieraz) < eps)
        for i in range(0, N):
            if per_bete[i] == True and per[i] == None and t > 10:
                per[i] = round(h*t/365.0, 3)

    print('Merkurioren periodoa ' + str(per[1]) + 's da.')
    print('Artizarraren periodoa ' + str(per[2]) + 's da.')
    print('Lurraren periodoa ' + str(per[3]) + 's da.')
    print('Marteren periodoa ' + str(per[4]) + 's da.')
    print('Jupiterren periodoa ' + str(per[5]) + 's da.')
    print('Saturnoren periodoa ' + str(per[6]) + 's da.')
    print('Uranoren periodoa ' + str(per[7]) + 's da.')
    print('Neptunoren periodoa ' + str(per[8]) + 's da.')
    print('Plutonen periodoa ' + str(per[9]) + 's da.\n\r')

    print('Exekuzio denbora ' + str(round(time.time() - exet, 2)) +
          's izan da.')
