"""
Created on Wed Oct 31 2018

@author: Song Bo (sbo@ualberta.ca)

This is a 3D Richards equation example simulating the small box.
"""

from __future__ import (print_function, division)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
import mpctools as mpc
from casadi import *
import matplotlib.pyplot as plt


start_time = time.time()
# ----------------------------------------------------------------------------------------------------------------------
# Define the geometry
# ----------------------------------------------------------------------------------------------------------------------
ratio_ft_to_m = 0.3048  # m/ft
lengthOfXinFt = 2  # feet
lengthOfYinFt = 4  # feet
lengthOfZinFt = 0.7874  # feet [In order to use 2 sensors of the probe, the minimum soil depth is 150+150+41=341mm or 1.11877ft)]
                         # The maximum height is 60.50 cm or 1.98 ft
lengthOfX = lengthOfXinFt * ratio_ft_to_m  # meter. Should be around 61 cm
lengthOfY = lengthOfYinFt * ratio_ft_to_m  # meter. Should be around 121.8 cm
# lengthOfZ = lengthOfZinFt * ratio_ft_to_m  # meter. Should be around 60.5 cm
lengthOfZ = 0.24

nodesInX = 12
nodesInY = 24
nodesInZ = 24
nodesInPlane = nodesInX*nodesInY
numberOfNodes = nodesInX*nodesInY*nodesInZ

dx = lengthOfX/nodesInX  # meter
dy = lengthOfY/nodesInY
dz = lengthOfZ/nodesInZ

# Label the nodes
positionOfNodes = []
for k in range(0, nodesInZ):
    for j in range(0, nodesInY):
        for i in range(0, nodesInX):
            positionOfNodes.append([i, j, k])

# Initial state
hIni = -0.01
hMatrix = hIni*ones(numberOfNodes)
x1 = zeros(numberOfNodes)
# head_pressure = zeros(numberOfNodes)
solList_h = []
thetaList = []
solList_theta = []
# sumTimeMatrix = 0.
# sumTimeSOR = 0
# orderOfTop = 1

# Parameters
Ks = 0.00000288889  # [m/s]
Theta_s = 0.43
Theta_r = 0.078
Alpha = 3.6
N = 1.56
S = 0.00001  # [per m]
PET = 0.0000000070042  # [per sec]


# Calculation of hydraulic conductivity
def hydraulic_conductivity(h, ks=Ks, alpha=Alpha, n=N):
    hc = 0.5*(((1+np.sign(h))*ks)+(1-np.sign(h))*ks*(((1+((-1*alpha*h)**n))**(-(1-1/n)))**(1/2))*
              ((1-(1-((1+((-1*alpha*h)**n))**(-(1-1/n)))**(n/(n-1)))**(1-1/n))**2))
    return hc


# Calculation of capillary capacity
def capillary_capacity(h, s=S, theta_s=Theta_s, theta_r=Theta_r, alpha=Alpha, n=N):
    c = 0.5*(((1+np.sign(h))*s)+(1-np.sign(h))*(s+((theta_s-theta_r)*alpha*n*(1-1/n))*((-1*alpha*h)**(n-1))*
                                                ((1+(-1*alpha*h)**n)**(-(2-1/n)))))
    return c


def mean_hydra_conductivity(left_boundary, right_boundary):
    lk = hydraulic_conductivity(left_boundary)
    rk = hydraulic_conductivity(right_boundary)
    mk = (lk+rk)/2
    return mk


# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def ode(x):
    head_pressure = SX.zeros(numberOfNodes)
    for i in range(0, numberOfNodes):
        state = x[i]
        coordinate = positionOfNodes[i]
        for index, item in enumerate(coordinate):
            if index == 0:
                if item == 0:
                    bc_xl = state
                    bc_xr = x[i+1]
                elif item == nodesInX-1:
                    bc_xl = x[i - 1]
                    bc_xr = state
                else:
                    bc_xl = x[i - 1]
                    bc_xr = x[i + 1]
            elif index == 1:
                if item == 0:
                    bc_yl = state
                    bc_yr = x[i + nodesInX]
                elif item == nodesInY-1:
                    bc_yl = x[i - nodesInX]
                    bc_yr = state
                else:
                    bc_yl = x[i - nodesInX]
                    bc_yr = x[i + nodesInX]
            else:
                if item == 0:
                    bc_zl = state
                    bc_zu = x[i + nodesInPlane]
                elif item == nodesInZ-1:
                    bc_zl = x[i - nodesInPlane]
                    KzU1 = hydraulic_conductivity(state)
                    bc_zu = state + dz*(-1 + 7.3999e-08/KzU1)
                else:
                    bc_zl = x[i - nodesInPlane]
                    bc_zu = x[i + nodesInPlane]

        KxL = mean_hydra_conductivity(x[i], bc_xl)
        KxR = mean_hydra_conductivity(x[i], bc_xr)
        deltaHxL = (x[i] - bc_xl) / dx
        deltaHxR = (bc_xr - x[i]) / dx

        KyL = mean_hydra_conductivity(x[i], bc_yl)
        KyR = mean_hydra_conductivity(x[i], bc_yr)
        deltaHyL = (x[i] - bc_yl) / dy
        deltaHyR = (bc_yr - x[i]) / dy

        KzL = mean_hydra_conductivity(x[i], bc_zl)
        KzU = mean_hydra_conductivity(x[i], bc_zu)
        deltaHzL = (x[i] - bc_zl) / dz
        deltaHzU = (bc_zu - x[i]) / dz

        temp0 = 1 / (0.5 * 2 * dx) * (KxR * deltaHxR - KxL * deltaHxL)
        temp1 = 1 / (0.5 * 2 * dy) * (KyR * deltaHyR - KyL * deltaHyL)
        temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
        temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
        temp4 = 0  # source term
        temp5 = temp0 + temp1 + temp2 + temp3 - temp4
        temp6 = temp5 / capillary_capacity(state)
        head_pressure[i] = temp6
    return head_pressure


# Time interval
KIni = hydraulic_conductivity(hIni)
CIni = capillary_capacity(hIni)
dt = 60  # second
# dt = 0.5*dz*dz/(KIni/CIni)*3
timeSpan = 720  # day
interval = int(timeSpan*60/dt)


# ode_casadi = mpc.getCasadiFunc(ode, [numberOfNodes], ['x'], funcname='ode')
model = mpc.DiscreteSimulator(ode, dt, [numberOfNodes], ['x'])

for t in range(0, interval):
    print('At day: ', t + 1)
    x1 = model(hMatrix)

    thetaList = []  # re-initialize thetaList
    for i in range(0, numberOfNodes):
        if x1[i] >= 0:
            theta = Theta_s
        else:
            theta = (Theta_s-Theta_r)*(1 + (-Alpha*x1[i])**N)**(-(1-1/N)) + Theta_r

        thetaList.append(theta)
    thetaList = np.asarray(thetaList)
    solList_theta.append(thetaList)

    # thetaList = (Theta_s-Theta_r)*(1 + (-Alpha*x1)**N)**(-(1-1/N)) + Theta_r
    # solList_theta.append(thetaList)

    hMatrix = copy(x1)
    solList_h.append(x1)

    # sumTimeMatrix += timeMatrix
    # sumTimeSOR += timeSOR
# print('Time elapsed for generating matrix: {:.3f}min'.format(sumTimeMatrix/60))
# print('Time elapsed for numerical calculation: {:.3f}min'.format(sumTimeSOR/60))
print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))
# print('{:.3f}% of time is used for generating matrix'.format(sumTimeMatrix/(time.time()-start_time)*100))