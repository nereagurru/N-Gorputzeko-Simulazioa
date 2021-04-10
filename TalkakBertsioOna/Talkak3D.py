# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:37:10 2021

@author: nerea
"""

import numpy as np
import random
import math
import sys


def talkaExis(N, D, x, y, z):
    """
       Talka jasango duten partikula bikoteak detektatu eta modu aleatorio
       batean partikula bikoteak ordenatzen dituen funtzioa.

    Jaso:
        N: partikula kopurua (int)
        D: partikulen diametroa (float)
        x: partikulen x koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        y: partikulen y koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        z: partikulen z koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)

    Itzuli:
        listOfCoordinates:  tuplaz osatutako lista, tuplako osagaiak talka
                            bikotean parte hartzen duten partikuleen indizeak
                            dira: [(part1, part2), (part3, part5),....]

    """

    # Diagonaleko elementuen arteko talka eta talka bakoitza bi aldiz ez kalkulatzeko
    eps = math.pow(10, -2)
    Rlagun = dist(N, x, y, z) + (D + eps)*np.tril(np.ones([N, N]))

    # talka gertatuko bada True gorde.
    talka = Rlagun < D

    # bere baitan True parametroa duten elementuen indizeak gorde tuplatan
    result = np.where(talka == True)
    listOfCoordinates = list(zip(result[0], result[1]))

    # tuplak ausaz berrordenatu
    random.shuffle(listOfCoordinates)
    return listOfCoordinates


def dist(N, x, y, z):
    """
        Partikulen arteko distantzia kalkulatzeko funtzioa.

    Jaso:
        N: partikula kopurua (int)
        x: partikulen x koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        y: partikulen y koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        z: partikulen z koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)

    Itzuli:
        R: partikulen arteko distantzia (NXN tamainako np array-n,
                                                    elementuak float)

    """

    xij = np.transpose(np.ones([N, N])*x) - np.ones([N, N])*x
    yij = np.transpose(np.ones([N, N])*y) - np.ones([N, N])*y
    zij = np.transpose(np.ones([N, N])*z) - np.ones([N, N])*z
    R = np.power(np.power(xij, 2) + np.power(yij, 2) + np.power(zij, 2), 0.5)
    return R


def talkaExekutatu(N, n, listOfCoor, x, y, z, vx, vy, vz):
    """
       Talka jasango duten partikula bikoteen lista  eta exekutatuko duen
       funtzioa hiru dimentsiotan masa berdina duten partikulentzat.

    Jaso:
        N: partikula kopurua (int)
        n: talka parametroa (float, [0,1] tarteko balio)
        x: partikulen x koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        y: partikulen y koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        z: partikulen z koordenatuko posizioak (N tamainako np array-n,
                                                elementuak float)
        vx: partikulen x koordenatuko abiadurak (N tamainako np array-n,
                                                elementuak float)
        vy: partikulen y koordenatuko abiadurak (N tamainako np array-n,
                                                elementuak float)
        vz: partikulen z koordenatuko abiadurak (N tamainako np array-n,
                                                elementuak float)

    Itzuli:
        vx: partikulen x koordenatuko abiadura berriak (N tamainako np array-n,
                                                        elementuak float)
        vy: partikulen y koordenatuko abiadura berriak (N tamainako np array-n,
                                                        elementuak float)
        vz: partikulen z koordenatuko abiadura berriak (N tamainako np array-n,
                                                        elementuak float)

    """

    # masa berdineko partikulak direnez, matrizea identikoa guztientzak
    matrix = np.array(([1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 1, 1], [1, -1, 0, 0, 0, 0],
                       [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]), dtype=float)

    # i iterazio bakoitzean talka bat exekutatuko da
    for i in range(0, len(listOfCoor)):

        # vx, vy eta vz behin baino gehiagotan indexatzea ekiditeko
        vxi = vx[listOfCoor[i][0]]
        vxj = vx[listOfCoor[i][1]]
        vyi = vy[listOfCoor[i][0]]
        vyj = vy[listOfCoor[i][1]]
        vzi = vz[listOfCoor[i][0]]
        vzj = vz[listOfCoor[i][1]]

        # Koordenatu bakoitzeko partikulen arteko distantzia kalkulatu
        rx = x[listOfCoor[i][0]] - x[listOfCoor[i][1]]
        ry = y[listOfCoor[i][0]] - y[listOfCoor[i][1]]
        rz = z[listOfCoor[i][0]] - z[listOfCoor[i][1]]

        # Talka ondorengo abiadura erlatiboak kalkulatu
        vxij, vyij, vzij = vErlatibo3D(n, rx, ry, rz, vxi - vxj, vyi - vyj,
                                       vzi-vzj)

        # Partikula bakoitzaren talka ondorengo abiadura kalkulatu
        b = np.array(([vxi + vxj, vyi + vyj, vzi + vzj, vxij, vyij, vzij]))
        sol = np.linalg.solve(matrix, b)

        # Talkako emaitzak gorde
        vx[listOfCoor[i][0]] = sol[0]
        vx[listOfCoor[i][1]] = sol[1]
        vy[listOfCoor[i][0]] = sol[2]
        vy[listOfCoor[i][1]] = sol[3]
        vz[listOfCoor[i][0]] = sol[4]
        vz[listOfCoor[i][1]] = sol[5]

    return vx, vy, vz


def talkaEbatzi(N, D, n, x, y, z, vx, vy, vz):
    listOfCoor = talkaExis(N, D, x, y, z)
    vxn, vyn, vzn = talkaExekutatu(N, n, listOfCoor, x, y, z, vx, vy, vz)
    return vxn, vyn, vzn


def vErlatibo3D(n, rx, ry, rz, vx, vy, vz):
    """
       Bi partikulen arteko talkako abiadura erlatiboa kalkulatzen
       duen funtzioa.

    Jaso:
        n: talka parametroa (float, [0,1] tarteko balio)
        rx: partikulen arteko x koordenatuko distantzia (float)
        ry: partikulen arteko y koordenatuko distantzia (float)
        rz: partikulen arteko z koordenatuko distantzia (float)
        vx: partikulen arteko x koordenatuko abiadura talka aurretik (float)
        vy: partikulen arteko y koordenatuko abiadura talka aurretik (float)
        vz: partikulen arteko z koordenatuko abiadura talka aurretik (float)

    Itzuli:
        vem[0]: partikulen arteko x koordenatuko abiadura talka ondoren (float)
        vem[1]: partikulen arteko y koordenatuko abiadura talka ondoren (float)
        vem[2]: partikulen arteko z koordenatuko abiadura talka ondoren (float)

    """

    # Normalizazio faktoreak
    rnor = math.pow(rx*rx + ry*ry + rz*rz, -0.5)
    tnor1 = math.pow(math.pow(rx + rz, 2) + 2*ry*ry, -0.5)
    tnor2 = math.pow(math.pow((rz*rx + ry*ry + rz*rz), 2) +
                     math.pow(rz*rx + ry*ry + rx*rx, 2) +
                     math.pow((rx - rz)*ry, 2), -0.5)

    # Abiadura normala eta tangentziala kalkulatu
    P = np.array(([rx*rnor, ry*rnor, rz*rnor], [tnor1*ry, -tnor1*(rx + rz),
                  tnor1*ry], [-tnor2*(rz*rx + ry*ry + rz*rz),
                  tnor2*(rx - rz)*ry, tnor2*(rz*rx + ry*ry + rx*rx)]))
    vem = np.array([vx, vy, vz])
    vkor = np.matmul(P, vem)

    # Osagai normala talka ondoren aldatu
    vkor[0] = -n*vkor[0]
    vem = np.matmul(P.T, vkor)

    return vem[0], vem[1], vem[2]
