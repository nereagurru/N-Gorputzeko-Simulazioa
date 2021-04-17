# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:34:42 2021

@author: nerea
"""


import numpy as np
import math
import matplotlib.pyplot as plt
import time


def axyz(N, G, D, m, x, y, z):
    """
        Funtzio honek partikula bakoitzaren grabitatearen ondoriozko
        azelerazioa kalkulatzen du 3 dimentsiotan.

    Jaso:
        N: partikula kopurua (int)
        G: konstante grabitazionala (float)
        D: partikulen diametroa (float)
        m: partikulen masak (N tamainako np array-n, elementuak float)
        x: partikulen x koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        y: partikulen y koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        z: partikulen z koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)

    Itzuli:
        ax: partikulen x koordenatuko azelerazioak (N tamainako np array-n,
                                                    elementuak float)
        ay: partikulen y koordenatuko azelerazioak (N tamainako np array-n,
                                                    elementuak float)
        az: partikulen z koordenatuko azelerazioak (N tamainako np array-n,
                                                    elementuak float)
    """
    # Partikulen arteko posizio erlatiboak kalkulatu
    xij = np.transpose(np.ones([N, N])*x) - np.ones([N, N])*x
    yij = np.transpose(np.ones([N, N])*y) - np.ones([N, N])*y
    zij = np.transpose(np.ones([N, N])*z) - np.ones([N, N])*z
    R = np.power(np.power(xij, 2) + np.power(yij, 2) + np.power(zij, 2), 0.5)

    # Zatitzaile nulurik ez izateko laguntzailea
    Rlagun = R + np.identity(N)

    # Talkarik ez badago bi elementuen artean batak besteari eragingo lioken
    # azelerazioa kalkulatzeko. Talka badago azelerazioa grabitatearen
    # ondorioz nulua
    matrixx = np.where(R > D, -G*np.transpose(m)*np.power(Rlagun, -3)*xij, 0)
    matrixy = np.where(R > D, -G*np.transpose(m)*np.power(Rlagun, -3)*yij, 0)
    matrixz = np.where(R > D, -G*np.transpose(m)*np.power(Rlagun, -3)*zij, 0)

    # Talka jasaten duten elementuen azelerazioa nulua
    eps = math.pow(10, -2)
    lagun = np.where(R+(D+eps)*np.identity(N) > D, True, False)
    ax = np.prod(lagun, axis=1)*np.sum(matrixx, axis=1)
    ay = np.prod(lagun, axis=1)*np.sum(matrixy, axis=1)
    az = np.prod(lagun, axis=1)*np.sum(matrixz, axis=1)

    return ax, ay, az
