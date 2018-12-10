"""
Created on Wed Oct 31 2018

@author: Song Bo (sbo@ualberta.ca)

This is a 3D Richards equation example simulating the small box.

ode is coded element by element
"""

from __future__ import (print_function, division)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import timeit
import csv
import mpctools as mpc
from casadi import *
import matplotlib.pyplot as plt


start_time = time.time()

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------








def Loam():
    pars = {}
    pars['thetaR'] = 0.078
    pars['thetaS'] = 0.43
    pars['alpha'] = 0.036*100
    pars['n'] = 1.56
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 1.04/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def LoamIni():
    pars = {}
    pars['thetaR'] = 0.078*0.9
    pars['thetaS'] = 0.43* 0.9
    pars['alpha'] = 0.036*100*0.9
    pars['n'] = 1.56* 0.9
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 1.04/100/3600*0.9
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def LoamOpt():
    pars = {}
    pars['thetaR'] = 6.18438096e-02
    pars['thetaS'] = 4.25461746e-01
    pars['alpha'] = 3.71675224e+00
    pars['n'] = 1.50828922e+00
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 2.99737827e-06
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def LoamySand():
    pars = {}
    pars['thetaR'] = 0.057
    pars['thetaS'] = 0.41
    pars['alpha'] = 0.124*100
    pars['n'] = 2.28
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 14.59/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def SandyLoam():
    pars = {}
    pars['thetaR'] = 0.065
    pars['thetaS'] = 0.41
    pars['alpha'] = 0.075*100
    pars['n'] = 1.89
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 4.42/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def SiltLoam():
    pars = {}
    pars['thetaR'] = 0.067
    pars['thetaS'] = 0.45
    pars['alpha'] = 0.020*100
    pars['n'] = 1.41
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 0.45/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def hFun(theta, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - pars['thetaR']) / (pars['thetaS'] - pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-pars['m'] + pars['mini']))
              - 1) + pars['mini']) ** (1. / (pars['n'] + pars['mini']))) / (-pars['alpha'] + pars['mini'])
    return psi


def thetaFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    theta = 100*(pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se)
    theta = theta.full()
    theta = theta.ravel(order='F')
    return theta


def KFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    K = pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/(pars['m']+pars['mini'])))+pars['mini'])**pars['m']+pars['mini'])**2
    K = K.full()
    K = K.ravel(order='F')
    return K


def CFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    dSedh=pars['alpha']*pars['m']/(1-pars['m']+pars['mini'])*(Se+pars['mini'])**(1/(pars['m']+pars['mini']))*(1-(Se+pars['mini'])**(1/(pars['m']+pars['mini']))+pars['mini'])**pars['m']
    C = Se*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh
    C = C.full().ravel(order='F')
    return C


def mean_hydra_conductivity(left_boundary, right_boundary):
    lk = KFun(left_boundary, pars)
    rk = KFun(right_boundary, pars)
    mk = (lk+rk)/2
    return mk


# Calculated the initial state
def ini_state_np(thetaIni, p):
    psiIni = hFun(thetaIni, p)
    # Approach: States in the same section are the same
    hMatrix = np.zeros(numberOfNodes)
    hMatrix[int(layersOfSensors[0] * nodesInPlane * ratio_z):int(layersOfSensors[1] * nodesInPlane * ratio_z)] = -5  # Top
    hMatrix[int(layersOfSensors[1] * nodesInPlane * ratio_z):int(layersOfSensors[2] * nodesInPlane * ratio_z)] = psiIni[0]
    hMatrix[int(layersOfSensors[2] * nodesInPlane * ratio_z):int(layersOfSensors[-1] * nodesInPlane * ratio_z)] = -3  # Bottom
    return hMatrix, psiIni


# Calculated the initial state
def ini_state_np_botToTop(thetaIni, p):
    hMatrix, psiIni = ini_state_np(thetaIni, p)
    hMatrix = hMatrix[::-1]
    psiIni = psiIni[::-1]
    return hMatrix, psiIni


def psiTopFun(hTop0, psi, qTop, dz):
    F = psi + 0.5*dz*(-1-(qTop+max(0, hTop0/dt))/KFun(hTop0, pars)) - hTop0  # Switching BC
    # F = psi + dz*(-1-(qTop)/KFun(hTop0, p)) - hTop0  # Only unsat. BC
    # F = psi + dz*(-1-(qTop-max(0, hTop0/dt))/KFun(hTop0, p)) - hTop0
    return F


# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def RichardsEQN_3D(x, t, u, u1):
    head_pressure = np.zeros(numberOfNodes)
    for i in range(0, numberOfNodes):
        dx = lengthOfX / nodesInX  # meter
        dy = lengthOfY / nodesInY
        dz = lengthOfZ / nodesInZ
        state = x[i]
        coordinate = positionOfNodes[i]
        for index, item in enumerate(coordinate):
            # print('Working with', i, 'node')
            if index == 0:
                if item == 0:
                    bc_xl = state
                    bc_xr = x[i+1]
                    dx = 0.5*dx
                elif item == nodesInX-1:
                    bc_xl = x[i - 1]
                    bc_xr = state
                    dx = 0.5*dx
                else:
                    bc_xl = x[i - 1]
                    bc_xr = x[i + 1]
            elif index == 1:
                if item == 0:
                    bc_yl = state
                    bc_yr = x[i + nodesInX]
                    dy = 0.5*dy
                elif item == nodesInY-1:
                    bc_yl = x[i - nodesInX]
                    bc_yr = state
                    dy = 0.5*dy
                else:
                    bc_yl = x[i - nodesInX]
                    bc_yr = x[i + nodesInX]
            else:
                if item == 0:
                    bc_zl = state
                    bc_zu = x[i + nodesInPlane]
                    dz = 0.5*dz
                elif item == nodesInZ-1:
                    bc_zl = x[i - nodesInPlane]
                    # KzU1 = hydraulic_conductivity(state)
                    # bc_zu = state + dz*(-1 - u/KzU1)
                    bc_zu = optimize.fsolve(psiTopFun, state, args=(state, u, dz))
                    dz = 0.5*dz
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
        temp6 = temp5 / CFun(state, pars)
        head_pressure[i] = temp6
    return head_pressure


def simulate(p):
    # define states and measurements arrays
    h = np.zeros(shape=(len(timeList), numberOfNodes))
    theta = np.zeros(shape=(len(timeList), numberOfNodes))
    h0, hIni = ini_state_np_botToTop(thetaIni, p)
    h[0] = h0
    theta0 = thetaFun(h0, p)  # Initial state of theta
    theta[0] = theta0[::-1]
    h_avg = np.zeros(shape=(len(timeList), numberOfSensors))  # 1 sensor
    h_avg[0] = hIni
    theta_avg = np.zeros(shape=(len(timeList), numberOfSensors))
    theta_avg[0] = thetaIni

    # Boundary conditions
    qTfun = irrigation
    for i in range(len(timeList)-1):  # in ts, end point is timeList[i+1], which is 2682*60
        print('From', i, ' min(s), to ', i+1, ' min(s)')
        ts = [timeList[i], timeList[i + 1]]
        if i == 414:
            pass

        y = integrate.odeint(RichardsEQN_3D, h0, ts, args=(qTfun[i], qTfun[i]))
        h0 = y[-1]
        h[i + 1] = h0

        theta0 = thetaFun(h0, p)
        theta0 = theta0[::-1]
        theta[i + 1] = theta0
        theta_avg[i+1] = np.matmul(CMatrix, theta0)
    return h_avg, theta_avg, theta, h


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Define the geometry
# ----------------------------------------------------------------------------------------------------------------------
ratio_x = 1
ratio_y = 1
ratio_z = 1

ratio_ft_to_m = 0.3048  # m/ft
lengthOfXinFt = 2  # feet
lengthOfYinFt = 4  # feet
lengthOfZinFt = 0.7874  # feet [In order to use 2 sensors of the probe, the minimum soil depth is 150+150+41=341mm or 1.11877ft)]
                         # The maximum height is 60.50 cm or 1.98 ft
lengthOfX = lengthOfXinFt * ratio_ft_to_m  # meter. Should be around 61 cm
lengthOfY = lengthOfYinFt * ratio_ft_to_m  # meter. Should be around 121.8 cm
# lengthOfZ = lengthOfZinFt * ratio_ft_to_m  # meter. Should be around 60.5 cm
lengthOfZ = 0.24

nodesInX = int(2*ratio_x)
nodesInY = int(2*ratio_y)
nodesInZ = int(24*ratio_z)

nodesInPlane = nodesInX*nodesInY
numberOfNodes = nodesInPlane*nodesInZ

dx = lengthOfX/nodesInX  # meter
dy = lengthOfY/nodesInY
dz = lengthOfZ/nodesInZ

# Label the nodes
positionOfNodes = []
for k in range(0, nodesInZ):
    for j in range(0, nodesInY):
        for i in range(0, nodesInX):
            positionOfNodes.append([i, j, k])

# ----------------------------------------------------------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------------------------------------------------------
numberOfSensors = 1
layersOfSensors = np.array([0, 5, 20, nodesInZ])  # beginning = top & end = bottom
# C matrix
start = layersOfSensors[1] * ratio_z
end = layersOfSensors[2] * ratio_z
difference = end - start
CMatrix = np.zeros((numberOfSensors, numberOfNodes))
for i in range(0, numberOfSensors):
    CMatrix[i][start * nodesInPlane: end * nodesInPlane] = 1. / ((end - start) * nodesInPlane)
    start += difference * ratio_z
    end += difference * ratio_z

# ----------------------------------------------------------------------------------------------------------------------
# Time interval
# ----------------------------------------------------------------------------------------------------------------------
ratio_t = 1
dt = 60.0*ratio_t  # second
timeSpan = 15
interval = int(timeSpan*60/dt)

timeList_original = np.arange(0, timeSpan+1)*dt/ratio_t

timeList = np.arange(0, interval+1)*dt

# ----------------------------------------------------------------------------------------------------------------------
# Inputs: irrigation
# ----------------------------------------------------------------------------------------------------------------------
irrigation = np.zeros(len(timeList))
for i in range(0, len(irrigation)):  # 1st node is constant for 1st temporal element. The last node is the end of the last temporal element.
    if i in range(0, 180):
        irrigation[i] = -0.050/86400
    elif i in range(180, 540):
        irrigation[i] = -0.010/86400
    else:
        irrigation[i] = 0

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
pars = Loam()
# pars_ini = LoamIni()
# pars_opt = LoamOpt()

# ----------------------------------------------------------------------------------------------------------------------
# Initial measurements
# ----------------------------------------------------------------------------------------------------------------------
thetaIni = array([21.6])  # left 21.6, right 18.6

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
h_i, theta_i, theta_i_all, h_i_all = simulate(pars)

# ----------------------------------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------------------------------
plt.figure()
# plt.plot(timeList_original/dt*ratio_t, theta_e[:, 0], 'b-.', label=r'$theta_1$ measured')
plt.plot(timeList/dt, theta_i[:, 0], 'y--', label=r'$theta_1$ initial')
# plt.plot(timeList/dt*ratio_t, theta_opt[:, 0], 'r--', label=r'$theta_1$ optimized')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
# plt.plot(timeList_original/dt*ratio_t, theta_e[:, 0], 'b-.', label=r'$theta_1$ measured')
plt.plot(timeList/dt, theta_i_all[:, 0], 'y--', label=r'$theta_1$ initial_Top')
# plt.plot(timeList/dt*ratio_t, theta_opt[:, 0], 'r--', label=r'$theta_1$ optimized')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))