"""
Created on Thu Aug 27 09:40:31 2018

@author: Song Bo (sbo@ualberta.ca)

This is a Richards equation parameter estimation example using orthogonal collocation on finite element.
dxdt = []

 = g(x,z,u,p)

The parameter to be estimated is

"""

# from __future__ import (print_function)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
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
def ini_state_MX(thetaIni, p, pars):
    psiIni = hFun(thetaIni, p, pars)

    hMatrix = MX.zeros(numberOfNodes)
    hMatrix[int(layersOfSensors[0] * nodesInPlane * ratio_z):int(layersOfSensors[1] * nodesInPlane * ratio_z)] = -5  # Top
    hMatrix[int(layersOfSensors[1] * nodesInPlane * ratio_z):int(layersOfSensors[2] * nodesInPlane * ratio_z)] = psiIni[0]
    hMatrix[int(layersOfSensors[2] * nodesInPlane * ratio_z):int(layersOfSensors[-1] * nodesInPlane * ratio_z)] = -3  # Bottom
    return hMatrix, psiIni


def ini_state_MX_botToTop(thetaIni, p, pars):
    hMatrix, psiIni = ini_state_MX(thetaIni, p, pars)
    hMatrix = hMatrix[::-1]
    psiIni = psiIni[::-1]
    return hMatrix, psiIni


def ini_state_np(thetaIni, p, pars):
    psiIni = hFun(thetaIni, p, pars)

    hMatrix = np.zeros(numberOfNodes)
    hMatrix[int(layersOfSensors[0] * nodesInPlane * ratio_z):int(layersOfSensors[1] * nodesInPlane * ratio_z)] = -5  # Top
    hMatrix[int(layersOfSensors[1] * nodesInPlane * ratio_z):int(layersOfSensors[2] * nodesInPlane * ratio_z)] = psiIni[0]
    hMatrix[int(layersOfSensors[2] * nodesInPlane * ratio_z):int(layersOfSensors[-1] * nodesInPlane * ratio_z)] = -3  # Bottom
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
#     F = psi[0] + 0.5*dz*(-1-(qTop+max(0, hTop0/h))/KFun(hTop0, pars)) - hTop0
#     # F = psi[0] + 0.5*dz*(-1-(qTop-max(0, hTop0/h))/KFun(hTop0, p, pars)) - hTop0
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


# Stage cost. Write in matrix form if not scalar. Currently in scalar
def stage_cost(yp, p, ym):
    thetaL = thetaFun_nofabs(yp, p, pars)
    theta_pred = mtimes(CMatrix, thetaL)
    difference = (theta_pred - ym)
    obj = difference**2
    obj = sum1(obj)  # sum column
    return obj


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
layersOfSensors_actual = np.array([0,4,19,nodesInZ])  # from bottom to top
# C matrix
start = layersOfSensors_actual[1] * ratio_z
end = layersOfSensors_actual[2] * ratio_z
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
timeSpan = 30
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
# Parameter estimation configuration
# ----------------------------------------------------------------------------------------------------------------------
pars = Loam()
# Parameters and initial guess
p_init = np.ones(5)       # Initial parameter guess
p_init[0] = p_init[0] * 0.9
p_init[1] = p_init[1] * 0.9
p_init[2] = p_init[2] * 0.9
p_init[3] = p_init[3] * 0.9
p_init[4] = p_init[4] * 0.9
p_min = p_init*0.6     # Parameter's lower bound
# p_min[1] = 0.365/pars['thetaS']  # Its a value smaller than 1
# p_min[2] = 0
p_max = p_init*1.3         # Parameter's upper bound
# p_max[1] = 10
# p_max[2] = 0.083/pars['thetaR']  # Its a value greater than 1

y_init = array([21.6])

# # Data. Size should match nk
# h_exp = np.zeros((len(timeList_original), 4))  # four sensors
# theta_exp = np.zeros((len(timeList_original), 4))
# with open('Data/exp_data_5L_4L_4361.dat', 'r') as f:
#     wholeFile = f.readlines()
#     for index, line in enumerate(wholeFile):
#         oneLine = line.rstrip().split(",")
#         oneLine = oneLine[1:5]
#         h_temp = []
#         theta_temp = []
#         for index1, item in enumerate(oneLine):
#             item = float(item)
#             theta_temp.append(item)
#             item = hFun(item, p_init, pars)
#             h_temp.append(item)
#         h_temp = array(h_temp, dtype='O')
#         theta_temp = array(theta_temp, dtype='O')
#         h_exp[index] = h_temp
#         theta_exp[index] = theta_temp
# y_exp = theta_exp      # Experimental measurments states/outputs
# u_exp = irrigation      # Experimental inputs

theta_exp = np.zeros((len(timeList_original), numberOfSensors))
with open('sim_results_1D_farm_robust', 'r') as f:
    wholeFile = f.readlines()
    for index, line in enumerate(wholeFile):
        oneLine = line.rstrip().split(" ")
        oneLine = array(oneLine, dtype='O')
        theta_exp[index] = oneLine
y_exp = theta_exp
u_exp = irrigation      # Experimental inputs

# ----------------------------------------------------------------------------------------------------------------------
# Generating the symbolic scheme
# ----------------------------------------------------------------------------------------------------------------------
# Dimensions
Nx = numberOfNodes                                                      # Number of differential states
Nz = 2*nodesInPlane                                                      # Number of algebraic states
Nu = 1                                                      # Number of inputs
Np = 5                                                      # Number of parameters to estimate

z = SX.sym("z", Nz)                                        # Algebraic state
x = SX.sym("x", Nx)                                        # Differential state
u = SX.sym("u", Nu)                                        # control
p = SX.sym("p", Np)                                         # parameters

# Create ODE and Objective functions
print("I am ready to create symbolic ODE")
ode = getODE(x, z, u, p, dz)
alg = getALG(x, z, u, p)
# f_q = stage_cost(x, k, xe)

# Create integrator
print("I am ready to create integrator")
dae = {'x':x, 'z':z, 'p':vertcat(u, p), 'ode':ode, 'alg':alg}
opts = {"tf":dt, "linear_solver":"csparse"} # interval length
I = integrator('I', 'idas', dae, opts)

# All parameter sets and irrigation amount
U = irrigation
P = MX.sym('P', Np)
G, G_ini = ini_state_MX_botToTop(y_init, P, pars)  # Initial state

# P = [1.,1.,1.,1.,1.]
# G, G_ini = ini_state_np(y_init, P, pars)  # Initial state

GL = []
GL.append(G)

Z = MX.zeros(nodesInPlane*2)
Z[0] = G[0]
Z[1] = G[1]
Z[2] = G[2]
Z[3] = G[3]
Z[4] = G[-4]
Z[5] = G[-3]
Z[6] = G[-2]
Z[7] = G[-1]

XE = y_exp

# Construct graph of integrator calls
print("I am ready to create construct graph")
J = 0  # Initial cost function
J += stage_cost(G, P, XE[0])  #  should be 0
for i in range(len(timeList)-1):
    Ik = I(x0=G, z0=Z, p=vertcat(U[i], P))  # integrator with initial state G, and input U[k]
    G = Ik['xf']  # Assign the finial state to the initial state
    Z = Ik['zf']
    # if ((i+1)*ratio_t) % 3 == 0:
    #     j = int((i+1)*ratio_t)
    #     J += stage_cost(G, P, XE[j])
    if ((i+1)*ratio_t) % 1 == 0:
        j = int((i+1)*ratio_t)
        J += stage_cost(G, P, XE[j])
    GL.append(G)


print("I am doing creating NLP solver function")
# Allocate an NLP solver
# nlp = {'x': P, 'f': J, 'g': vertcat(*GL)}  # x: Solve for P (parameters), which gives the lowest J (cost fcn), with the constraints G (propagated model)
nlp = {'x': P, 'f': J, 'g': G}  # x: Solve for P (parameters), which gives the lowest J (cost fcn), with the constraints G (propagated model)
# opts = {"ipopt.linear_solver":"ma97"}
# opts = {"ipopt.linear_solver": "ma57"}
opts = {"ipopt.linear_solver": "mumps"}
# opts["ipopt.hessian_approximation"] = 'limited-memory'
opts["ipopt.print_level"] = 5
opts["ipopt.jacobian_approximation"] = "finite-difference-values"
opts["regularity_check"] = True
opts["verbose"] = True
# opts["ipopt.acceptable_tol"] = 1e-05;
# opts["ipopt.tol"]=1e-05
opts['ipopt.max_iter'] = 100
print("I am ready to build")
solver = nlpsol('solver', 'ipopt', nlp, opts)

print("I am ready to solve")

sol = solver(
        lbx=p_min,
        ubx=p_max,
        x0=p_init  # Initial guess of decision variable
      )
print (sol)

pe = sol['x'].full().squeeze()
pe[0] = pe[0]*pars['Ks']
pe[1] = pe[1]*pars['thetaS']
pe[2] = pe[2]*pars['thetaR']
pe[3] = pe[3]*pars['alpha']
pe[4] = pe[4]*pars['n']
print ("")
print ("Estimated parameter(s) is(are): " + str(pe))
p_init[0] = p_init[0]*pars['Ks']
p_init[1] = p_init[1]*pars['thetaS']
p_init[2] = p_init[2]*pars['thetaR']
p_init[3] = p_init[3]*pars['alpha']
p_init[4] = p_init[4]*pars['n']
print ("")
print ("Actual value(s) is(are): " + str(p_init))

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))
# np.savetxt('sim_results_1D_farm_white_noise', rand_e)
# # io.savemat('sim_results_1D_odeint.mat', dict(y_1D_odeint=theta_i))
