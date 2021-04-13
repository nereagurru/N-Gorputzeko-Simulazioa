# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:37:10 2021

@author: nerea
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt


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
        listOfCoor: tuplaz osatutako lista, tuplako osagaiak talka
                    bikotean parte hartzen duten partikuleen indizeak
                    dira: [(part1, part2), (part3, part5),....]
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
    """
        Partikulen arteko talka detektatu eta exekutatzen duen funtzioa. Kasu
        honetan partikulek masa eta diametro bera dute.
    Jaso:
        N: partikula kopurua (int)
        D: partikulen diametroa (N tamainako np array-n, elementuak float)
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
    listOfCoor = talkaExis(N, D, x, y, z)
    vx, vy, vz = talkaExekutatu(N, n, listOfCoor, x, y, z, vx, vy, vz)
    return vx, vy, vz


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


def energiaZinetiko(vx, vy, vz):
    v2 = vx*vx + vy*vy + vz*vz
    return np.sum(v2)


if __name__ == "__main__":
    """
    Demagun D=1 diametroko eta m=1 masako zortzi partikula ditugula l=10
    aldeko kutxa batean. Partikula bakoitzaren zentroa kutxaren ertz batean
    kokatzen da. Hasierako baldintza modura t=0 aldiunean partikula guztiak
    kutxaren zentrorantz higituko dira abiaduraren modulu berdinarekin.
    Helburua talkak deskribatzeko kodea frogatzea denez, partikulen arteko
    elkarrekintza grabitatorioak arbuiatuko dira. n=1 denean energia zinetikoa
    kontserbatzen denez gero, edozein aldiunetan partikula bakoitzaren
    abiaduraren moduluaren karratuaren batura totala berdina izan behar da.
    
    N = 8
    D = 1.0
    n = 0
    x = np.array([5, -5, -5, 5, 5, -5, -5, 5], dtype=float)
    y = np.array([5, 5, -5, -5, 5, 5, -5, -5], dtype=float)
    z = np.array([-5, -5, -5, -5, 5, 5, 5, 5], dtype=float)
    vx = np.array([-0.9, 0.6, 0.4, -0.8, -1, 1.6, 1.2, -0.9], dtype=float)
    vy = np.array([-1.3, -0.1, 1.4, 1.5, -1.9, -1.7, 1, 0.7], dtype=float)
    vz = np.array([1.8, 0.3, 0.7, 1, -0.2, -0.7, -0.4, -1.6], dtype=float)

    fig = plt.figure()
    ardatz = fig.add_subplot(111)

    plt.ylabel(r'$Energia$')
    plt.xlabel('t')
    #plt.ylim((23.99998, 24.00002))
    vmzx = np.sum(vx)
    vmzy = np.sum(vy)
    vmzz = np.sum(vz)
    print('Masa zentroaren energia zinetikoa ' + str((vmzx*vmzx + vmzy*vmzy + vmzz*vmzz)))
    print('Hasierako egoeraren energia zinetikoa ' + str(energiaZinetiko(vx, vy, vz)))
    loop = 1000
    h = 0.01
    for t in range(1, loop):
        if t*h > 0 and t*h < 10:
            plt.scatter(t*h, energiaZinetiko(vx, vy, vz), color = 'purple')

        vx, vy, vz = talkaEbatzi(N, D, n, x, y, z, vx, vy, vz)
        x = x + vx*h
        y = y + vy*h
        z = z + vz*h
    print('Amaierako egoeraren energia zinetikoa ' + str(energiaZinetiko(vx, vy, vz)))
    #plt.savefig(fname='EnergiaZinetikoa_n=0,9.png')
    plt.show()
    """
    N = 2
    D = 1.0
    n1 = 1
    n2 = 0.9
    n3 = 0.4
    n4 = 0
    x = np.array([5, -5], dtype=float)
    y = np.array([0, 0], dtype=float)
    z = np.array([0, 0], dtype=float)
    vx = np.array([-1, 5], dtype=float)
    vy = np.array([0, 0], dtype=float)
    vz = np.array([0, 0], dtype=float)
    
    
    x1 = x.copy()
    y1 = y.copy()
    z1 = z.copy()
    vx1 = vx.copy()
    vy1 = vy.copy()
    vz1 = vz.copy()
    
    x2 = x.copy()
    y2 = y.copy()
    z2 = z.copy()
    vx2 = vx.copy()
    vy2 = vy.copy()
    vz2 = vz.copy()
    
    x3 = x.copy()
    y3 = y.copy()
    z3 = z.copy()
    vx3 = vx.copy()
    vy3 = vy.copy()
    vz3 = vz.copy()
    
    x4 = x.copy()
    y4 = y.copy()
    z4 = z.copy()
    vx4 = vx.copy()
    vy4 = vy.copy()
    vz4 = vz.copy()

    fig = plt.figure()
    ardatz = fig.add_subplot(111)

    plt.ylabel(r'$Energia$')
    plt.xlabel('t')

    fig.text(0.3, 0.95, "Masa zentroa", ha="center", va="bottom", size="large", color='red')
    fig.text(0.45, 0.95, "n=1", ha="center", va="bottom", size="large",color="purple")
    fig.text(0.55, 0.95, "n=0.9", ha="center", va="bottom", size="large",color="blue")
    fig.text(0.65, 0.95, "n=0.4", ha="center", va="bottom", size="large",color="green")
    fig.text(0.75, 0.95, "n=0", ha="center", va="bottom", size="large",color="yellow")
    #plt.ylim((23.99998, 24.00002))
    vmzx = np.sum(vx)
    vmzy = np.sum(vy)
    vmzz = np.sum(vz)
    print('Masa zentroaren energia zinetikoa ' + str((vmzx*vmzx + vmzy*vmzy + vmzz*vmzz)/2))
    print('Hasierako egoeraren energia zinetikoa ' + str(energiaZinetiko(vx, vy, vz)))
    loop = 2000
    h = 0.001
    for t in range(1, loop):       
        if t*h > 1.4 and t*h < 1.8:
            if t%10 == 0: 
                plt.scatter(t*h, (vmzx*vmzx + vmzy*vmzy + vmzz*vmzz)/2, color = 'red')
                plt.scatter(t*h, energiaZinetiko(vx1, vy1, vz1), color = 'purple')
                if t%15 == 0:
                    plt.scatter(t*h, energiaZinetiko(vx2, vy2, vz2), color = 'blue')
                if t%20 == 0:
                    plt.scatter(t*h, energiaZinetiko(vx4, vy4, vz4), color = 'yellow')
                if t%30 == 0:                    
                    plt.scatter(t*h, energiaZinetiko(vx3, vy3, vz3), color = 'green')
        vx1, vy1, vz1 = talkaEbatzi(N, D, 1, x1, y1, z1, vx1, vy1, vz1)
        x1 = x1 + vx1*h
        y1 = y1 + vy1*h
        z1 = z1 + vz1*h
        
        vx2, vy2, vz2 = talkaEbatzi(N, D, n2, x2, y2, z2, vx2, vy2, vz2)
        x2 = x2 + vx2*h
        y2 = y2 + vy2*h
        z2 = z2 + vz2*h
        
        vx3, vy3, vz3 = talkaEbatzi(N, D, n3, x3, y3, z3, vx3, vy3, vz3)
        x3 = x3 + vx3*h
        y3 = y3 + vy3*h
        z3 = z3 + vz3*h
        
        vx4, vy4, vz4 = talkaEbatzi(N, D, n4, x4, y4, z4, vx4, vy4, vz4)
        x4 = x4 + vx4*h
        y4 = y4 + vy4*h
        z4 = z4 + vz4*h

    print('Amaierako egoeraren energia zinetikoa ' + str(energiaZinetiko(vx, vy, vz)))
    plt.savefig(fname='EnergiaZinetikoak.png')
    plt.show()
