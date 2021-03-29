# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:39:21 2021

@author: nerea
"""

import numpy as np
import random
import math
import sys

def talkaExis(D, N, x, y):
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
    Rlagun = dist(N, x, y) + (D + eps)*np.tril(np.ones([N, N])) 


    talka = Rlagun < D  #talka gertatuko bada True gorde. 
    #talkaren eragina kalkulatzeko modua aleatorioa izan behar da.
    #Hmen posible ahal da paralelizazioa sortzea? grabitatea kalkulatzea alde 
    #batetik behar dutenentzat, eta besteentzat talkak
    
    result = np.where(talka==True)
    listOfCoordinates = list(zip(result[0], result[1]))
    random.shuffle(listOfCoordinates)
    return talka, listOfCoordinates


def dist(N, x, y):
    xij = np.transpose(np.ones([N, N])*x) - np.ones([N, N])*x  # xi - xj
    yij = np.transpose(np.ones([N, N])*y) - np.ones([N, N])*y
    R = np.power(np.power(xij, 2) + np.power(yij, 2), 0.5)
    return R


def talkaExekutatu(N, n, listOfCoor, x, y, vx, vy):
    """
       2D-tan talka 
    """
    for i in range(0, len(listOfCoor)):
        
        matrix = np.array(([1, 1, 0, 0], [0, 0, 1, 1],
                           [1, -1, 0, 0], [0, 0, 1, -1]), dtype=float)
        vxx = vx[listOfCoor[i][0]] - vx[listOfCoor[i][1]]
        vyy = vy[listOfCoor[i][0]] - vy[listOfCoor[i][1]]
        rx = x[listOfCoor[i][0]] - x[listOfCoor[i][1]]
        ry = y[listOfCoor[i][0]] - y[listOfCoor[i][1]]
        vxij, vyij = vErlatibo3D(n, rx, ry, vxx, vyy)
        b = np.array(([vx[listOfCoor[i][0]] + vx[listOfCoor[i][1]], 
                       vy[listOfCoor[i][0]] + vy[listOfCoor[i][1]], 
                       vxij, vyij]))
    
        sol = np.linalg.solve(matrix, b)
        vx[listOfCoor[i][0]] = sol[0]
        vx[listOfCoor[i][1]] = sol[1]
        vy[listOfCoor[i][0]] = sol[2]
        vy[listOfCoor[i][1]] = sol[3]

    return vx, vy


def talkaEbatzi(N, D, n, x, y, vx, vy):
    talka, listOfCoor = talkaExis(D, N, x, y)
    vxn, vyn = talkaExekutatu(N, n, listOfCoor, x, y, vx, vy)
    return vxn, vyn

def vErlatibo2D(n, rx, ry, vx, vy):
    rnor = math.pow(rx*rx + ry*ry, -0.5)
    
    # Normalizazioan arazoa egon daiteke rx, ry edo rz = 0 direnean.
    # Hau konpontzeko, suposatuko da iñoiz ez dela existituko rx=ry=rz=0 kasua
    

    P = np.array(([rx*rnor, ry*rnor],[-rnor*ry, rnor*rx]))
    vem = np.array([vx, vy])
    vkor = np.matmul(P, vem)
    vkor[0] = -n*vkor[0]
    vem = np.matmul(P.T, vkor)
    
    return vem[0], vem[1]


if __name__ == "__main__":
    aber = time.time()
    N = 2
    D = 3
    n = 0.3
    x = np.array([-1, 1], dtype=float)
    y = np.array([-1, 1], dtype=float)
    vx = np.array([1, -1], dtype=float)
    vy = np.array([1, -2], dtype=float)
    
    print("Hasieran :")
    print("vx = " + str(vx))
    print("vy = " + str(vy))
    vx, vy = talkaEbatzi(N, D, n, x, y, vx, vy)
    print("Talka eta gero :")
    print("vx = " + str(vx))
    print("vy = " + str(vy))
    print("Programa honek " + str(time.time()-aber) + " s behar izan ditu.")
