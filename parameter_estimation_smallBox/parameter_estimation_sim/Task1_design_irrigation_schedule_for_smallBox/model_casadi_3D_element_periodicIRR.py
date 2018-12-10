"""
Created on Thu Nov 01 2018

@author: Song Bo (sbo@ualberta.ca)

This is a 3D finite difference model of Richards equation.

The model is symbolic and solved using solvers in idas/cvodes

odes are coded element by element
"""

from __future__ import (print_function, division)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import timeit
import csv
from casadi import *
import matplotlib.pyplot as plt


print("I am starting up")
start_time = time.time()
# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
def Loam():
    pars = {}
    pars['thetaR'] = 0.078
    pars['thetaS'] = 0.43
    pars['alpha'] = 0.036 * 100
    pars['n'] = 1.56
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 1.04 / 100 / 3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001  # same as RichardsFoam2
    pars['mini'] = 1.e-20
    return pars


def Loam_opt():
    pars = {}
    pars['thetaR'] = 6.84574617e-02
    pars['thetaS'] = 4.04711688e-01
    pars['alpha'] = 3.26708465e+00
    pars['n'] = 1.49503601e+00
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 2.57868116e-06
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001  # same as RichardsFoam2
    pars['mini'] = 1.e-20
    return pars


def LoamySand():
    pars = {}
    pars['thetaR'] = 0.057
    pars['thetaS'] = 0.41
    pars['alpha'] = 0.124 * 100
    pars['n'] = 2.28
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 14.59 / 100 / 3600
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
# General functions
# ----------------------------------------------------------------------------------------------------------------------
def hFun(theta, p, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100. - p[2]*pars['thetaR']) / (p[1]*pars['thetaS'] - p[2]*pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-(1-1/(p[4]*pars['n']+pars['mini'])) + pars['mini']))
              - 1) + pars['mini']) ** (1. / (p[4]*pars['n'] + pars['mini']))) / (-p[3]*pars['alpha'] + pars['mini'])
    return psi


def thetaFun(psi, p, pars):
    Se = if_else(psi>=0., 1., (1+SX.fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    theta = 100*(p[2]*pars['thetaR']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*Se)
    # theta = theta.full().ravel()
    return theta


def thetaFun_nofabs(psi, p, pars):
    Se = if_else(psi>=0., 1., (1+(((psi**2)**0.5)*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    theta = 100.*(p[2]*pars['thetaR']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*Se)
    # theta = theta.full().ravel()
    return theta


def KFun(psi, p, pars):
    Se = if_else(psi>=0., 1., (1+SX.fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    K = p[0]*pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])**2
    # K = K.full().ravel()
    return K


def CFun(psi, p, pars):
    Se = if_else(psi>=0., 1., (1+SX.fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    dSedh=p[3]*pars['alpha']*(1-1/(p[4]*pars['n']+pars['mini']))/(1-(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])*(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini']))*(1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini']))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))
    C = Se*pars['Ss']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*dSedh
    # C = C.full().ravel()
    return C


# ----------------------------------------------------------------------------------------------------------------------
# tanh functions
# ----------------------------------------------------------------------------------------------------------------------
def hFun_tanh(theta, p, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100. - p[2]*pars['thetaR']) / (p[1]*pars['thetaS'] - p[2]*pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-(1-1/(p[4]*pars['n']+pars['mini'])) + pars['mini']))
              - 1) + pars['mini']) ** (1. / (p[4]*pars['n'] + pars['mini']))) / (-p[3]*pars['alpha'] + pars['mini'])
    return psi


def thetaFun_tanh(psi, p, pars):
    Se = 0.5*((1+sign(psi))*1.+(1-sign(psi))*((1+SX.fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini'])))))
    theta = 100*(p[2]*pars['thetaR']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*Se)
    # theta = theta.full().ravel()
    return theta


def KFun_tanh(psi, p, pars):
    Se = 0.5*((1+sign(psi))*1.+(1-sign(psi))*((1+SX.fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini'])))))
    K = p[0]*pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])**2
    # K = K.full().ravel()
    return K


def CFun_tanh(psi, p, pars):
    Se = 0.5*((1+sign(psi))*1.+(1-sign(psi))*((1+SX.fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini'])))))
    dSedh=p[3]*pars['alpha']*(1-1/(p[4]*pars['n']+pars['mini']))/(1-(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])*(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini']))*(1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini']))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))
    C = Se*pars['Ss']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*dSedh
    # C = C.full().ravel()
    return C


# Calculated the initial state
def ini_state(thetaIni, p, pars):
    psiIni = hFun(thetaIni, p, pars)

    hMatrix = np.zeros(numberOfNodes)
    hMatrix[int(layersOfSensors[0] * nodesInPlane * ratio_z):int(layersOfSensors[1] * nodesInPlane * ratio_z)] = psiIni[0]  # Top
    hMatrix[int(layersOfSensors[1] * nodesInPlane * ratio_z):int(layersOfSensors[2] * nodesInPlane * ratio_z)] = psiIni[0]
    hMatrix[int(layersOfSensors[2] * nodesInPlane * ratio_z):int(layersOfSensors[-1] * nodesInPlane * ratio_z)] = psiIni[0]  # Bottom
    return hMatrix, psiIni


def ini_state_np(thetaIni, p, pars):
    psiIni = hFun(thetaIni, p, pars)

    hMatrix = np.zeros(numberOfNodes)
    hMatrix[int(layersOfSensors[0] * nodesInPlane * ratio_z):int(layersOfSensors[1] * nodesInPlane * ratio_z)] = psiIni[0]  # Top
    hMatrix[int(layersOfSensors[1] * nodesInPlane * ratio_z):int(layersOfSensors[2] * nodesInPlane * ratio_z)] = psiIni[0]
    hMatrix[int(layersOfSensors[2] * nodesInPlane * ratio_z):int(layersOfSensors[-1] * nodesInPlane * ratio_z)] = psiIni[0]  # Bottom
    return hMatrix, psiIni


def ini_state_np_botToTop(thetaIni, p, pars):
    hMatrix, psiIni = ini_state_np(thetaIni, p, pars)
    hMatrix = hMatrix[::-1]
    psiIni = psiIni[::-1]
    return hMatrix, psiIni


def ini_state_interp(thetaIni4, p, pars):
    psiIni4 = hFun(thetaIni4, p, pars)
    d1 = (thetaIni4[1] - thetaIni4[0])/2.  # RHS - LHS
    firstState = np.array([thetaIni4[0] - d1])
    d2 = (thetaIni4[-1] - thetaIni4[-2])*5/6  # RHS - LHS
    lastState = np.array([thetaIni4[-1] + d2])

    thetaIni5 = np.append(firstState, thetaIni4)
    thetaIni6 = np.append(thetaIni5, lastState)

    xp = [0, 4, 11, 18, 25, 31]
    x_array = np.arange(0, 32)
    thetaIni32 = np.interp(x_array, xp, thetaIni6)

    psiIni32 = hFun(thetaIni32, p, pars)

    return psiIni32, psiIni4


# def psiTopFun(hTop0, p, psi, qTop, dz):
#     # F = psi[0] + 0.5*dz*(-1-(qTop+max(0, hTop0/h))/KFun(hTop0, pars)) - hTop0
#     F = psi[0] + 0.5*dz*(-1-(qTop-max(0, hTop0/h))/KFun(hTop0, p, pars)) - hTop0
#     return F


def getODE(x, z, u, p, dz):
    # psiBot = z[0]
    # psiTop = z[1]
    dhdt = SX.zeros(numberOfNodes)
    zBotIndex = 0
    zTopIndex = nodesInPlane
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
                    bc_zl = z[zBotIndex]
                    bc_zu = x[i + nodesInPlane]
                    dz = 0.5*dz
                    zBotIndex+=1
                elif item == nodesInZ-1:
                    bc_zl = x[i - nodesInPlane]
                    bc_zu = z[zTopIndex]
                    dz = 0.5*dz
                    zTopIndex+=1
                else:
                    bc_zl = x[i - nodesInPlane]
                    bc_zu = x[i + nodesInPlane]

        KxL = (KFun(x[i], p, pars) + KFun(bc_xl, p, pars))/2
        KxR = (KFun(x[i], p, pars) + KFun(bc_xr, p, pars))/2
        deltaHxL = (x[i] - bc_xl) / dx
        deltaHxR = (bc_xr - x[i]) / dx

        KyL = (KFun(x[i], p, pars) + KFun(bc_yl, p, pars))/2
        KyR = (KFun(x[i], p, pars) + KFun(bc_yr, p, pars))/2
        deltaHyL = (x[i] - bc_yl) / dy
        deltaHyR = (bc_yr - x[i]) / dy

        KzL = (KFun(x[i], p, pars) + KFun(bc_zl, p, pars))/2
        KzU = (KFun(x[i], p, pars) + KFun(bc_zu, p, pars))/2
        deltaHzL = (x[i] - bc_zl) / dz
        deltaHzU = (bc_zu - x[i]) / dz

        temp0 = 1 / (0.5 * 2 * dx) * (KxR * deltaHxR - KxL * deltaHxL)
        temp1 = 1 / (0.5 * 2 * dy) * (KyR * deltaHyR - KyL * deltaHyL)
        temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
        temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
        temp4 = 0  # source term
        temp5 = temp0 + temp1 + temp2 + temp3 - temp4
        temp6 = temp5 / CFun(state, p, pars)
        dhdt[i] = temp6
    return dhdt


def getALG(x, z, u, p):
    # xBot = z[0]
    # xTop = z[1]
    res = [
        z[0] - x[0],
        z[1] - x[1],
        z[2] - x[2],
        z[3] - x[3],
        z[4] - (x[-4] + 0.5 * dz * (-1 - (u + SX.fmax(z[4] / dt, 0)) / KFun(z[4], p, pars))),
        z[5] - (x[-3] + 0.5 * dz * (-1 - (u + SX.fmax(z[5] / dt, 0)) / KFun(z[5], p, pars))),
        z[6] - (x[-2] + 0.5 * dz * (-1 - (u + SX.fmax(z[6] / dt, 0)) / KFun(z[6], p, pars))),
        z[7] - (x[-1] + 0.5 * dz * (-1 - (u + SX.fmax(z[7] / dt, 0)) / KFun(z[7], p, pars)))
           ]
    # res = [xTop - (x[0]+0.5*dz*(-1-(u-SX.fmax(xTop/h, 0))/KFun(xTop, p, pars))),
    #        xBot - (x[-1]+dz/2.)]
    return vertcat(*res)


# ----------------------------------------------------------------------------------------------------------------------
# Model information, setup and cost function
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

numberOfSensors = 1

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
layersOfSensors = np.array([0,5,20,nodesInZ])  # beginning = top & end = bottom
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
timeSpan = 2880
interval = int(timeSpan*60/dt)

timeList_original = np.arange(0, timeSpan+1)*dt/ratio_t

timeList = np.arange(0, interval+1)*dt

# ----------------------------------------------------------------------------------------------------------------------
# Inputs: irrigation
# ----------------------------------------------------------------------------------------------------------------------
# The idea here is to irrigate the water periodically
irrigation = np.zeros(len(timeList))
# for i in range(0, len(irrigation)):  # 1st node is constant for 1st temporal element. The last node is the end of the last temporal element.
#     if i in range(0, 180):   # lets say 0 means 6am. 6am-9am
#         irrigation[i] = 0.
#     elif i in range(180, 360):  # 9am - 12pm
#         irrigation[i] = -0.050/86400
#     elif i in range(360, 540):  # 12pm - 3pm
#         irrigation[i] = -0.050/86400
#     elif i in range(540, 720):  # 3pm - 6pm
#         irrigation[i] = -0.050/86400
#     else:
#         irrigation[i] = 0.
for i in range(0, len(irrigation)):  # 1st node is constant for 1st temporal element. The last node is the end of the last temporal element.
    if i in range(0, 180):   # lets say 0 means 6am. 6am-9am
        irrigation[i] = 0.
    elif i in range(180, 240):  # 9am - 10pm
        irrigation[i] = -0.20/86400
    elif i in range(240, 300):  # 10am - 11pm
        irrigation[i] = -0.00/86400
    elif i in range(300, 360):  # 11am - 12pm
        irrigation[i] = -0.20/86400
    elif i in range(360, 420):  # 12pm - 1pm
        irrigation[i] = -0.00/86400
    elif i in range(420, 480):  # 1pm - 2pm
        irrigation[i] = -0.20 / 86400
    elif i in range(480, 540):  # 2pm - 3pm
        irrigation[i] = -0.00 / 86400
    else:
        irrigation[i] = 0.

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
pars = Loam_opt()

# ----------------------------------------------------------------------------------------------------------------------
# Initial state
# ----------------------------------------------------------------------------------------------------------------------
y_init = array([15.])

# ----------------------------------------------------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------------------------------------------------
# # # Data. Size should match nk
# # h_exp = np.zeros((len(timeList_original), 4))  # four sensors
# # theta_exp = np.zeros((len(timeList_original), 4))
# # with open('Data/exp_data_5L_4L_4361.dat', 'r') as f:
# #     wholeFile = f.readlines()
# #     for index, line in enumerate(wholeFile):
# #         oneLine = line.rstrip().split(",")
# #         oneLine = oneLine[1:5]
# #         h_temp = []
# #         theta_temp = []
# #         for index1, item in enumerate(oneLine):
# #             item = float(item)
# #             theta_temp.append(item)
# #             item = hFun(item, p_init, pars)
# #             h_temp.append(item)
# #         h_temp = array(h_temp, dtype='O')
# #         theta_temp = array(theta_temp, dtype='O')
# #         h_exp[index] = h_temp
# #         theta_exp[index] = theta_temp
# # y_exp = theta_exp      # Experimental measurments states/outputs
# # u_exp = irrigation      # Experimental inputs
# #
# theta_exp = np.zeros((len(timeList_original), numberOfSensors))
# with open('sim_results_1D_farm_robust', 'r') as f:
#     wholeFile = f.readlines()
#     for index, line in enumerate(wholeFile):
#         oneLine = line.rstrip().split(" ")
#         oneLine = array(oneLine, dtype='O')
#         theta_exp[index] = oneLine
# y_exp = theta_exp
# u_exp = irrigation      # Experimental inputs

# ----------------------------------------------------------------------------------------------------------------------
# Generating the symbolic scheme
# ----------------------------------------------------------------------------------------------------------------------
# Dimensions
Nx = numberOfNodes                                                      # Number of differential states
Nz = 2*nodesInPlane                                                      # Number of algebraic states
Nu = 1                                                      # Number of inputs
Np = 5                                                      # Number of parameters to estimate

# Declare variables (use scalar graph)
x = SX.sym("x", Nx)                                        # Differential state
z = SX.sym("z", Nz)                                        # Algebraic state
u = SX.sym("u", Nu)                                        # control
p = SX.sym("p", Np)                                         # parameters

# Create ODE and Objective functions
ode = getODE(x, z, u, p, dz)                                     # Symbolic ode
alg = getALG(x, z, u, p)                                     # Symbolic alg

dae = {'x':x, 'z':z, 'p':vertcat(u, p), 'ode':ode, 'alg':alg}

# Setup plant for simulation
opts = {"tf":dt, "linear_solver":"csparse"} # interval length
I = integrator('I', 'idas', dae, opts)

# # ----------------------------------------------------------------------------------------------------------------------
# # Initial guess parameters
# # ----------------------------------------------------------------------------------------------------------------------
# p_init = np.ones(5)       # Initial parameter guess
# p_init[0] = p_init[0] * 0.9
# p_init[1] = p_init[1] * 0.9
# p_init[2] = p_init[2] * 0.9
# p_init[3] = p_init[3] * 0.9
# p_init[4] = p_init[4] * 0.9
# G, G4 = ini_state_np_botToTop(y_init, p_init, pars)
# GL = np.zeros((len(timeList), numberOfNodes))
# GL[0] = G
# Z = [G[0], G[1], G[2], G[3], G[-4], G[-3], G[-2], G[-1]]
#
# for i in range(len(timeList)-1):
#     print('From', i, ' min(s), to ', i + 1, ' min(s)')
#     if i == 414:
#         pass
#     Ik = I(x0=G, z0=Z, p=vertcat(irrigation[i], p_init))  # integrator with initial state G, and input U[k]
#     G = Ik['xf']  # Assign the finial state to the initial state
#     G_np = Ik['xf'].full().ravel()
#     GL[i+1] = G_np
#     Z = Ik['zf']
#     # Z[0] = G[0]
# thetaL = thetaFun_nofabs(GL, p_init, pars)
# thetaL = thetaL.full()
# theta_i = np.matmul(thetaL, CMatrix.T)
# # difference_i = theta_i - theta_exp
# # obj_i = difference_i**2
# # obj_i_sum = sum1(obj_i)
# # obj_i_sum = sum2(obj_i_sum)
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Optimized results
# # ----------------------------------------------------------------------------------------------------------------------
# p_opt = np.array([2.43906930e-06/pars['Ks'], 3.99958425e-01/pars['thetaS'], 6.55643757e-02/pars['thetaR'],
#                   3.17212079e+00/pars['alpha'], 1.48375362e+00/pars['n']])
# G_opt, G4_opt = ini_state_np_botToTop(y_init, p_opt, pars)
# GL_opt = np.zeros((len(timeList), numberOfNodes))
# GL_opt[0] = G_opt
# Z_opt = [G_opt[0], G_opt[1], G_opt[2], G_opt[3], G_opt[-4], G_opt[-3], G_opt[-2], G_opt[-1]]
#
# for i in range(len(timeList)-1):
#     print('From', i, ' min(s), to ', i + 1, ' min(s)')
#     if i == 414:
#         pass
#     Ik = I(x0=G_opt, z0=Z_opt, p=vertcat(irrigation[i], p_opt))  # integrator with initial state G_opt, and input U[k]
#     G_opt = Ik['xf']  # Assign the finial state to the initial state
#     G_np_opt = Ik['xf'].full().ravel()
#     GL_opt[i+1] = G_np_opt
#     Z_opt = Ik['zf']
#     # Z_opt[0] = G_opt[0]
# thetaL_opt = thetaFun_nofabs(GL_opt, p_opt, pars)
# thetaL_opt = thetaL_opt.full()
# theta_opt = np.matmul(thetaL_opt, CMatrix.T)
# # difference_i = theta_opt - theta_exp
# # obj_i = difference_i**2
# # obj_i_sum = sum1(obj_i)
# # obj_i_sum = sum2(obj_i_sum)

# ----------------------------------------------------------------------------------------------------------------------
# Experimental parameters
# ----------------------------------------------------------------------------------------------------------------------
p_exp = np.ones(5)       # Initial parameter guess
G, G4 = ini_state_np_botToTop(y_init, p_exp, pars)
GL = np.zeros((len(timeList), numberOfNodes))
GL[0] = G
Z = [G[0], G[1], G[2], G[3], G[-4], G[-3], G[-2], G[-1]]

for i in range(len(timeList)-1):
    print('From', i, ' min(s), to ', i + 1, ' min(s)')
    if i == 414:
        pass
    Ik = I(x0=G, z0=Z, p=vertcat(irrigation[i], p_exp))  # integrator with initial state G, and input U[k]
    G = Ik['xf']  # Assign the finial state to the initial state
    G_np = Ik['xf'].full().ravel()
    GL[i+1] = G_np
    Z = Ik['zf']
    # Z[0] = G[0]
thetaL_exp = thetaFun_nofabs(GL, p_exp, pars)
thetaL_exp = thetaL_exp.full()
for index, item in enumerate(thetaL_exp):
    thetaL_exp[index] = item[::-1]
theta_exp = np.matmul(thetaL_exp, CMatrix.T)
# difference_i = theta_exp - theta_exp
# obj_i = difference_i**2
# obj_i_sum = sum1(obj_i)
# obj_i_sum = sum2(obj_i_sum)

plt.figure()
# plt.plot(timeList_original/dt*ratio_t, y_exp[:, 0], 'b-.', label=r'$\theta_{1, exp}$')
plt.plot(timeList_original/dt*ratio_t, theta_exp[:, 0], 'y--', label=r'$\theta_{1, exp}$')
# plt.plot(timeList/dt*ratio_t, theta_i[:, 0], 'y-', label=r'$\theta_{1, ini}$')
# plt.plot(timeList/dt*ratio_t, theta_opt[:, 0], 'r:', label=r'$\theta_{1, opt}$')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
# plt.plot(timeList_original/dt*ratio_t, theta_e[:, 0], 'b-.', label=r'$theta_1$ measured')
plt.plot(timeList/dt, thetaL_exp[:, 0], 'y--', label=r'$\theta_1$ initial_Top')
# plt.plot(timeList/dt*ratio_t, theta_opt[:, 0], 'r--', label=r'$theta_1$ optimized')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

np.savetxt('sim_results_1D_farm_robust', theta_exp)
# io.savemat('sim_results_1D_farm_robust', dict(y_1D_farm=theta_e))