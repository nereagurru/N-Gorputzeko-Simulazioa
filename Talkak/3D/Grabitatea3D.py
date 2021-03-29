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
    Grabitatearen ondorioz elkarrekintzak jasatzen dituzten partikulen
    azelerazioa kalkulatzeko funtzioa 2D-tan.

    ax, ay itzuli
    """
    xij = np.transpose(np.ones([N, N])*x) - np.ones([N, N])*x  # xi - xj
    yij = np.transpose(np.ones([N, N])*y) - np.ones([N, N])*y
    zij = np.transpose(np.ones([N, N])*z) - np.ones([N, N])*z
    R = np.power(np.power(xij, 2) + np.power(yij, 2) + np.power(zij, 2), 0.5)  # R
    Rlagun = R + np.identity(N)
    matrixx = np.where(R>D, -G*m*np.power(Rlagun, -3)*xij , 0) 
    matrixy = np.where(R>D, -G*m*np.power(Rlagun, -3)*yij, 0)
    matrixz = np.where(R>D, -G*m*np.power(Rlagun, -3)*zij, 0)

    return np.sum(matrixx, axis=1), np.sum(matrixy, axis=1), np.sum(matrixz, axis=1)
