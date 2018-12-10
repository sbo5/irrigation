from __future__ import (print_function)
import numpy as np
from numpy import diag, zeros, dot, copy
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm

# Define the geometry
lengthOfX = 0.5  # meter  # 1.0 instead of 1: because 1.0 can make it be a float instead of an integer
lengthOfY = 0.5
lengthOfZ = 1.

nodesInX = 10
nodesInY = 10
nodesInZ = 50
nodesInPlane = nodesInX*nodesInY
numberOfNodes = nodesInX*nodesInY*nodesInZ

dx = lengthOfX/nodesInX  # meter
dy = lengthOfY/nodesInY
dz = lengthOfZ/nodesInZ

# Label the nodes
indexOfNodes = []
for i in range(0, numberOfNodes):
    indexOfNodes.append(i)
positionOfNodes = []
for k in range(0, nodesInZ):
    for j in range(0, nodesInY):
        for i in range(0, nodesInX):
            positionOfNodes.append([i, j, k])

# Time interval
dt = 86400  # second

# Initial state
hMatrix = -0.01*np.ones(numberOfNodes)
KMatrix = []
solList_h = []
thetaList = []
solList_theta = []

# Parameters
Ks = 0.00000288889  # [m/s]
Theta_s = 0.43
Theta_r = 0.078
Alpha = 3.6
N = 1.56
S = 0.00001  # [per m]


# Calculation of hydraulic conductivity
def hydraulic_conductivity(h, ks=Ks, alpha=Alpha, n=N):
    if h < 0:
        temp000 = ks
        temp001 = (-1*alpha*h)**n
        temp002 = 1 + temp001
        temp003 = temp002**(-(1 - (1/n)))
        temp004 = temp003**(1/2)
        temp005 = 1 - temp003**(n/(n-1))  # double check
        temp006 = 1 - temp005**(1-(1/n))
        temp007 = temp006**2
        kk = temp000*temp004*temp007
    else:
        kk = ks
    return kk


# Calculation of capillary capacity
def capillary_capacity(h, s=S, theta_s=Theta_s, theta_r=Theta_r, alpha=Alpha, n=N):
    if h < 0:
        temp00 = (theta_s - theta_r)*alpha*n*(1 - 1/n)
        temp01 = -1*alpha*h
        temp02 = temp01**(n-1)
        temp03 = 1 + temp01**n
        temp04 = temp03**(-(2 - 1/n))
        c = s + temp00*temp02*temp04
    else:
        c = s
    return c


for t in range(0, 4):
    # A matrix
    A = np.zeros((numberOfNodes, numberOfNodes))
    for i in range(0, numberOfNodes-1):  # Upper diagonal +1
        if i % nodesInX != (nodesInX-1):  # not the last point in x-direction
            KxR1 = hydraulic_conductivity(hMatrix[i])
            KxR2 = hydraulic_conductivity(hMatrix[i+1])
            KxR = np.mean([KxR1, KxR2])  # the algorithm of calculating the average can be changed
            A[i, i+1] = -1/(0.5*2*dx*dx)*KxR  # When it is not equally spaced, cannot use this formula
        else:  # boundary condition
            KxR1 = hydraulic_conductivity(hMatrix[i])
            KxR2 = hydraulic_conductivity(hMatrix[i])  # b.c. zero gradient
            KxR = np.mean([KxR1, KxR2])  # the algorithm of calculating the average can be changed
            A[i, i+1] = -1/(0.5*2*dx*dx)*KxR  # When it is not equally spaced, cannot use this formula
    for i in range(1, numberOfNodes):  # Lower diagonal -1
        if i % nodesInX != 0:  # not the first point in x-direction
            KxL1 = hydraulic_conductivity(hMatrix[i])
            KxL2 = hydraulic_conductivity(hMatrix[i-1])
            KxL = np.mean([KxL1, KxL2])
            A[i, i-1] = -1/(0.5*2*dx*dx)*KxL
        else:
            KxL1 = hydraulic_conductivity(hMatrix[i])
            KxL2 = hydraulic_conductivity(hMatrix[i])  # b.c. zero gradient
            KxL = np.mean([KxL1, KxL2])
            A[i, i-1] = -1/(0.5*2*dx*dx)*KxL
    j = 0
    k = 0
    for i in range(0, numberOfNodes-nodesInX):  # Upper diagonal +3 (+nodesInX)
        if i >= nodesInX*(nodesInY-1) + nodesInPlane*j:
            KyR1 = hydraulic_conductivity(hMatrix[i])
            KyR2 = hydraulic_conductivity(hMatrix[i])  # b.c.
            KyR = np.mean([KyR1, KyR2])
            A[i, i+nodesInX] = -1/(0.5*2*dy*dy)*KyR
            k += 1
            if k % nodesInX == 0:
                j += 1

        else:
            KyR1 = hydraulic_conductivity(hMatrix[i])
            KyR2 = hydraulic_conductivity(hMatrix[i+nodesInX])
            KyR = np.mean([KyR1, KyR2])
            A[i, i+nodesInX] = -1/(0.5*2*dy*dy)*KyR
    k = 0
    j = 0
    for i in range(nodesInX, numberOfNodes):  # Lower diagonal -3 (-nodesInX)
        if i-nodesInX >= nodesInX*(nodesInY-1) + nodesInPlane*j:
            KyL1 = hydraulic_conductivity(hMatrix[i])
            KyL2 = hydraulic_conductivity(hMatrix[i])
            KyL = np.mean([KyL1, KyL2])
            A[i, i-nodesInX] = -1/(0.5*2*dy*dy)*KyL
            k += 1
            if k % nodesInX == 0:
                j += 1
        else:
            KyL1 = hydraulic_conductivity(hMatrix[i])
            KyL2 = hydraulic_conductivity(hMatrix[i-nodesInX])
            KyL = np.mean([KyL1, KyL2])
            A[i, i-nodesInX] = -1/(0.5*2*dy*dy)*KyL

    for i in range(0, numberOfNodes - nodesInPlane):  # Upper diagonal +9
        KzU1 = hydraulic_conductivity(hMatrix[i])
        KzU2 = hydraulic_conductivity(hMatrix[i+nodesInPlane])
        KzU = np.mean([KzU1, KzU2])
        A[i, i+nodesInPlane] = -1/(0.5*2*dz*dz)*KzU  # for a more general case, dx should be replaced by (x_()-x_())
    for i in range(nodesInPlane, numberOfNodes):  # Lower diagonal -9
        KzL1 = hydraulic_conductivity(hMatrix[i])
        KzL2 = hydraulic_conductivity(hMatrix[i-nodesInPlane])
        KzL = np.mean([KzL1, KzL2])
        A[i, i-nodesInPlane] = -1/(0.5*2*dz*dz)*KzL  # for a more general case, dx should be replaced by (x_()-x_())

    for i in range(0, numberOfNodes):  # Main diagonal
        summation = sum(A[i])
        A[i, i] = -summation + capillary_capacity(hMatrix[i])/dt

    # B matrix
    B = np.zeros(numberOfNodes)
    for i in range(0, numberOfNodes):
        temp0 = capillary_capacity(hMatrix[i])*hMatrix[i]/dt
        temp1 = 0  # AET-source term
        if i in range(0, nodesInPlane):  # lowest layer
            KzL1b = hydraulic_conductivity(hMatrix[i])
            KzL2b = hydraulic_conductivity(hMatrix[i])  # due to the lower boundary - free drainage
            KzLb = np.mean([KzL1b, KzL2b])

            KzU1b = hydraulic_conductivity(hMatrix[i])
            KzU2b = hydraulic_conductivity(hMatrix[i + nodesInPlane])
            KzUb = np.mean([KzU1b, KzU2b])

            temp2 = (KzUb - KzLb)/(0.5*2*dz)
            B[i] = temp0 + temp2 - temp1
        elif i in range(numberOfNodes - nodesInPlane, numberOfNodes):  # highest layer
            KzL1b = hydraulic_conductivity(hMatrix[i])
            KzL2b = hydraulic_conductivity(hMatrix[i - nodesInPlane])
            KzLb = np.mean([KzL1b, KzL2b])

            KzU1b = hydraulic_conductivity(hMatrix[i])                               # double check with Soumya
            KzU2b = hydraulic_conductivity(hMatrix[i] + dz*(-1 + 7.3999e-08/KzU1b))  # due to the top boundary
            # KzU2b = hydraulic_conductivity(1/3*(-hMatrix[i-nodesInPlane] + 4*hMatrix[i] + 2*dz*(-1 - 7.3999e-08 / KzU1b)))
            KzUb = np.mean([KzU1b, KzU2b])

            temp2 = (KzUb-KzLb)/(0.5*52*dz)
            B[i] = temp0 + temp2 - temp1
        else:
            KzL1b = hydraulic_conductivity(hMatrix[i])
            KzL2b = hydraulic_conductivity(hMatrix[i - nodesInPlane])
            KzLb = np.mean([KzL1b, KzL2b])

            KzU1b = hydraulic_conductivity(hMatrix[i])
            KzU2b = hydraulic_conductivity(hMatrix[i + nodesInPlane])  # due to the top boundary
            KzUb = np.mean([KzU1b, KzU2b])

            temp2 = (KzUb-KzLb)/(0.5*2*dz)
            B[i] = temp0 + temp2 - temp1
    # AMatrix = np.savetxt('AMatrix', A)
    # BMatrix = np.savetxt('BMatrix', B)
    A_inv = inv(A)
    # Ai = inv(A_inv)  # check if A is well-conditioned by testing if A_ive == Ai
    # conditionNumber = cond(A)
    # rank = matrix_rank(A)
    # positiveDefinite = cholesky(A)
    hSol = dot(A_inv, B)

    #  SOR
    omega = 1.9  # Relaxation factor
    nMax = 3*numberOfNodes  # Number of iterations
    en = zeros(nMax)
    rn = zeros(nMax)  # used for plot
    jn = zeros(nMax)  # used for plot
    varEps = 1.0e-6

    d = diag(A)

    x0 = copy(hMatrix)
    r0 = B - dot(A, x0)
    # r0 = B - dot(A, hMatrix)
    x1 = copy(x0)
    k = 0
    for i in range(0, nMax):
        for j in range(0, numberOfNodes):
            a = A[j]
            r0[j] = B[j] - dot(a, x1)
            x1[j] = x0[j] + omega*(r0[j]/d[j])

        ERE = abs((x0 - x1)/x0)  # 1 criteria
        RAbs = abs(r0)  # 2nd criteria

        en[i] = norm(ERE)
        rn[i] = norm(RAbs)  # used for plot
        jn[i] = i  # used for plot
        if float(norm(RAbs)) < varEps:
            if float(norm(ERE)) < varEps:
                print('solution was obtained in ', i, 'iterations')
                k = 1
                break
        x0 = copy(x1)  # after break, these two following line are not executed
        print('iteration: ', i)
    if k == 1:
        print('solution obtained')
    else:
        print('solution was not obtained within nMax iterations!')

    thetaList = []  # re-initialize thetaList
    for i in range(0, numberOfNodes):
        if x1[i] >= 0:
            theta = copy(Theta_s)
        else:
            theta = (Theta_s-Theta_r)*(1 + (-Alpha*x1[i])**N)**(-(1-1/N))

        thetaList.append(theta)
    thetaList = np.asarray(thetaList)
    solList_theta.append(thetaList)

    hMatrix = copy(x1)
    solList_h.append(x1)
