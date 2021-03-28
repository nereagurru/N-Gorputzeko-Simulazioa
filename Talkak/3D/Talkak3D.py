# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:37:10 2021

@author: nerea
"""

import numpy as np
import random
import math
import sys

def talkaExis(D, N, x, y, z):
    """
       Talka gertatuko den partikula bikoteak detektatu eta modu aleatorio
       batean partikula bikoteak ordenatu.
       D partikula baten diametroa
       N partikula kopurua
       R N*N numpy objektua partikula bikoteen distantzia adierazten duena
       output1: talka gertatuko den edo ez adieraziko duen boolearra
       output2: tuplaz osatutako lista, [(part1, part2), (part3, part5),....]
       
    """
    #Diagonaleko elementuen arteko talka kontuan ez hartzeko, ezta talka bakoitza
    #bi aldiz ere
    eps = math.pow(10, -2)
    Rlagun = dist(N, x, y, z) + (D + eps)*np.tril(np.ones([N, N])) 


    talka = Rlagun < D  #talka gertatuko bada True gorde. 
    #talkaren eragina kalkulatzeko modua aleatorioa izan behar da.
    #Hmen posible ahal da paralelizazioa sortzea? grabitatea kalkulatzea alde 
    #batetik behar dutenentzat, eta besteentzat talkak
    
    result = np.where(talka==True)
    listOfCoordinates = list(zip(result[0], result[1]))
    random.shuffle(listOfCoordinates)
    return talka, listOfCoordinates


def dist(N, x, y, z):
    xij = np.transpose(np.ones([N, N])*x) - np.ones([N, N])*x  # xi - xj
    yij = np.transpose(np.ones([N, N])*y) - np.ones([N, N])*y
    zij = np.transpose(np.ones([N, N])*z) - np.ones([N, N])*z
    R = np.power(np.power(xij, 2) + np.power(yij, 2) + np.power(zij, 2), 0.5)
    return R


def talkaExekutatu(N, n, listOfCoor, x, y, z, vx, vy, vz):
    """
       2D-tan talka 
    """
    for i in range(0, len(listOfCoor)):
        matrix = np.array(([1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1],
                           [1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]))
        vxx = vx[listOfCoor[i][0]] - vx[listOfCoor[i][1]]
        vyy = vy[listOfCoor[i][0]] - vy[listOfCoor[i][1]]
        vzz = vz[listOfCoor[i][0]] - vz[listOfCoor[i][1]]
        rx = x[listOfCoor[i][0]] - x[listOfCoor[i][1]]
        ry = y[listOfCoor[i][0]] - y[listOfCoor[i][1]]
        rz = z[listOfCoor[i][0]] - z[listOfCoor[i][1]]
        vxij, vyij, vzij = vErlatibo3D(n, rx, ry, rz, vxx, vyy, vzz)
        b = np.array(([vx[listOfCoor[i][0]] + vx[listOfCoor[i][1]], 
                       vy[listOfCoor[i][0]] + vy[listOfCoor[i][1]], 
                       vz[listOfCoor[i][0]] + vz[listOfCoor[i][1]],
                       vxij, vyij, vzij]))
        sol = np.linalg.solve(matrix, b)
        vx = sol[0:2]
        vy = sol[2:4]
        vz = sol[4:6]
        
    return vx, vy, vz


def talkaEbatzi(N, D, n, x, y, z, vx, vy, vz):
    talka, listOfCoor = talkaExis(D, N, x, y, z)
    vxn, vyn, vzn = talkaExekutatu(N, n, listOfCoor, x, y, z, vx, vy, vz)
    return vxn, vyn, vzn


def vErlatibo3D(n, rx, ry, rz, vx, vy, vz):
    rnor = math.pow(rx*rx + ry*ry + rz*rz, -0.5)
    
    # Normalizazioan arazoa egon daiteke rx, ry edo rz = 0 direnean.
    # Hau konpontzeko, suposatuko da iñoiz ez dela existituko rx=ry=rz=0 kasua
    
    tnor1 = math.pow(math.pow(rx + rz, 2) + 2*ry*ry, -0.5)
    tnor2 = math.pow(math.pow((rz*rx + ry*ry + rz*rz), 2) + math.pow(rz*rx + ry*ry + rx*rx, 2) + math.pow((rx - rz)*ry, 2) , -0.5)
    P = np.array(([rx*rnor, ry*rnor, rz*rnor],[tnor1*ry, -tnor1*(rx + rz), tnor1*ry],[-tnor2*(rz*rx + ry*ry + rz*rz), tnor2*(rx - rz)*ry, tnor2*(rz*rx + ry*ry + rx*rx)]))
    vem = np.array([vx, vy, vz])
    vkor = np.matmul(P, vem)
    vkor[0] = -n*vkor[0]
    vem = np.matmul(P.T, vkor)
    
    return vem[0], vem[1], vem[2]

if __name__ == "__main__":
    N = 2
    n = 1
    x = np.array([0, -1])
    y = np.array([0, -1])
    z = np.array([0.0, 0.0])
    vx = np.array([0, 1])
    vy = np.array([0, 1])
    vz = np.array([0.0, 0.0])
    print("Hasieran :")
    print("vx = " + str(vx))
    print("vy = " + str(vy))
    vx, vy, vz = talkaEbatzi(N, 3, n, x, y, z, vx, vy, vz)
    print("Talka eta gero :")
    print("vx = " + str(vx))
    print("vy = " + str(vy))
    print("vz = " + str(vz))
