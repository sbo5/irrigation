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
def ini_state(thetaIni, p, pars):
    psiIni = hFun(thetaIni, p, pars)

    hMatrix = np.zeros(numberOfNodes)
    hMatrix[0*ratio_z:9*ratio_z] = psiIni[0]  # 1st section has 8 states
    hMatrix[9*ratio_z:16*ratio_z] = psiIni[1]  # After, each section has 7 states
    hMatrix[16*ratio_z:23*ratio_z] = psiIni[2]
    hMatrix[23*ratio_z:numberOfNodes] = psiIni[3]

    # hMatrix = np.zeros(numberOfNodes)
    # hMatrix[int(0):int(1)] = psiIni[0]  # 1st section has 8 states
    # hMatrix[int(1):int(2)] = psiIni[1]  # After, each section has 7 states
    # hMatrix[int(2):int(3)] = psiIni[2]
    # hMatrix[int(3):numberOfNodes] = psiIni[3]

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
    psiTop = z[0]
    psiBot = z[1]

    q = SX.sym('q', numberOfNodes+1, 1)
    # Bottom boundary
    KBot = KFun(psiBot, p, pars)
    q[-1] = -KBot * ((x[-1] - psiBot) / dz * 2 + 1.0)
    # Top boundary
    KTop = KFun(psiTop, p, pars)
    q[0] = -KTop * ((psiTop - x[0]) / dz * 2 + 1)

    C = CFun(x, p, pars)
    i = np.arange(0, numberOfNodes - 1)
    Knodes = KFun(x, p, pars)
    Kmid = (Knodes[i] + Knodes[i + 1]) / 2

    j = np.arange(1, numberOfNodes)
    q[j] = -Kmid * ((x[i] - x[i + 1]) / dz + 1.)

    i = np.arange(0, numberOfNodes)
    dhdt = (-(q[i] - q[i + 1]) / dz) / C
    return dhdt


def getALG(x, z, u, p):
    xTop = z[0]
    xBot = z[1]
    res = [xTop - (x[0]+0.5*dz*(-1-(u+SX.fmax(xTop/h, 0))/KFun(xTop, p, pars))),
           xBot - x[-1]]
    # res = [xTop - (x[0]+0.5*dz*(-1-(u-SX.fmax(xTop/h, 0))/KFun(xTop, p, pars))),
    #        xBot - (x[-1]+dz/2.)]
    return vertcat(*res)


# Stage cost. Write in matrix form if not scalar. Currently in scalar
def stage_cost(yp, ym):
    return mtimes((yp-ym).T,(yp-ym))


# Function to convert states to measured outputs. Do for your system appropriately.
def getOutputs(x, z, u, p):
    y = thetaFun(x, p, pars)
    y_pred = mtimes(CMatrix, y)
    return y_pred


# ----------------------------------------------------------------------------------------------------------------------
# Model information, setup and cost function
# ----------------------------------------------------------------------------------------------------------------------
# First section: define the geometry
ratio_z = 1
lengthOfZ = 0.67  # meter
nodesInZ = int(32*ratio_z)  # Define the nodes
intervalInZ = nodesInZ  # The distance between the boundary to its adjoint states is 1/2 of delta_z
nodesInPlane = 1
numberOfNodes = nodesInZ*nodesInPlane
dz = lengthOfZ/intervalInZ
numberOfSensors = 4

# Second section: define time interval
# tf = 261600.0                                        # End time
# nk = 4360                                          # Number of data points/finite elements
# tf = 160920.0
# nk = 2682
# tf = 86580.0
# nk = 1443
tf = 86400.0/2
nk = 720
# tf = 120.
# nk = 2
ratio_t = 1
h = (tf/nk)*ratio_t                                        # Size of the finite elements

default_time_interval = 60  # seconds
interval = int(nk*default_time_interval/h)
timeList_original = np.arange(0, nk+1)*h/ratio_t
timeList = np.arange(0, interval+1)*h

# Model parameters
r = .22

# Top boundaries
irrigation = np.zeros(len(timeList))
for i in range(0, len(irrigation)):  # 1st node is constant for 1st temporal element. The last node is the end of the last temporal element.
    # # 4361 case
    # if i in range(0, int(22/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * r * r) / (22 * 60))
    # elif i in range(int(59/ratio_t), int(87/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * r * r) / (27 * 60))
    # elif i in range(int(161/ratio_t), int(189/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * r * r) / (27 * 60))
    # elif i in range(int(248/ratio_t), int(276/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * r * r) / (27 * 60))
    # elif i in range(int(335/ratio_t), int(361/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * r * r) / (25 * 60))
    # elif i in range(int(1590/ratio_t), int(1656/ratio_t)):
    #     irrigation[i] = -(0.004 / (pi * r * r) / (65 * 60))
    # else:
    #     irrigation[i] = 0
    # # 2683 case
    # if i in range(int(1/ratio_t), int(1216/ratio_t)):
    # if i in range(int(1150/ratio_t), int(1216/ratio_t)):
    #     irrigation[i] = -(0.004 / (pi * r * r) / (65 * 60))
    # else:
    #     irrigation[i] = 0
    # # 1000 case
    # if i in range(int(40/ratio_t), int(106/ratio_t)):
    #     irrigation[i] = -(0.004 / (pi * r * r) / (65 * 60))
    # else:
    #     irrigation[i] = 0

    if i in range(180, 540):
        irrigation[i] = -0.010/86400
    else:
        irrigation[i] = 1.0e-20

# ---------------------------------------------------------------------
# Parameter estimation configuration
# ---------------------------------------------------------------------
# Estimation parameters
c = 3                                           # Number of collocation points (start and end points included)
d = c-1                                          # Degree of interpolating polynomial
cost_type = 0                               # 0 = SSE (sum of squared error) or 1 = ASSE (Average of SSE)
use_jit = 1                                 # 0 for No, 1 for Yes. Use JIT (Just-in-time) compiler for added speed. For small problems JIT does not make a difference.
pars = Loam()

# Bounds on variables
# Parameters and initial guess
p_init = np.ones(5)       # Initial parameter guess
p_min = np.ones(5)*0.9     # Parameter's lower bound
# p_min[1] = 0.365/pars['thetaS']  # Its a value smaller than 1
# p_min[2] = 0
p_max = np.ones(5)*1.1         # Parameter's upper bound
# p_max[1] = 10
# p_max[2] = 0.083/pars['thetaR']  # Its a value greater than 1

# Differential state bounds and initial guess
x_min = np.ones(nodesInZ)*(-inf)         # Lower bound on states
x_max = np.zeros(nodesInZ)      # Upper bound on states

# y_init = np.array([10.1, 8.4, 8.6, 10.0])  # 1444 case
# y_init = array([30.2, 8.8, 8.7, 10.0])  # 2683 case: left to right = top to bottom
y_init = array([30., 30., 30., 30.])

x_init, x_init4 = ini_state(y_init, p_init, pars)       # Initial state guess
xi_min = x_init      # Initial condition/state [1]
xi_max = x_init      # Initial condition/state. Same value as [1]

# Algebraic state bounds and initial guess
z_min = np.array([-inf, -inf])           # Lower bound on states
z_max = np.array([inf, inf])            # Upper bound on states
z_init = np.array([x_init[0], x_init[-1]])          # Initial state guess

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

theta_exp = np.zeros((721, 4))
with open('sim_results_1D_farm_robust', 'r') as f:
    wholeFile = f.readlines()
    for index, line in enumerate(wholeFile):
        oneLine = line.rstrip().split(" ")
        oneLine = array(oneLine, dtype='O')
        theta_exp[index] = oneLine
y_exp = theta_exp
u_exp = irrigation      # Experimental inputs

# C matrix
start = 2 * ratio_z
end = 9 * ratio_z
CMatrix = np.zeros((numberOfSensors, numberOfNodes))
for i in range(0, 4):
    if i == 0:
        CMatrix[i][(start - 1 * ratio_z) * nodesInPlane: end * nodesInPlane] = 1. / (
                    (end - (start - 1 * ratio_z)) * nodesInPlane)
    else:
        CMatrix[i][start * nodesInPlane: end * nodesInPlane] = 1. / ((end - start) * nodesInPlane)
    start += 7 * ratio_z
    end += 7 * ratio_z
# CMatrix = np.eye(4)

# -----------------------------------------------------------------------------------
# Generating the symbolic scheme
# -----------------------------------------------------------------------------------

# Dimensions
Nx = 32                                                      # Number of differential states
# Nx = 4
Nz = 2                                                      # Number of algebraic states
Nu = 1                                                      # Number of inputs
Np = 5                                                      # Number of parameters to estimate
Ny = 4                                                      # Number of outputs

# Declare variables (use scalar graph)
t = SX.sym("t")                                            # time

xdot = SX.sym("xdot", Nx)                                   # xdot
x = SX.sym("x", Nx)                                        # Differential state
z = SX.sym("z", Nz)                                        # Algebraic state
u = SX.sym("u", Nu)                                        # control
p = SX.sym("p", Np)                                         # parameters

ym = SX.sym("ym", Ny)                                       # measured outputs
yp = SX.sym("yp", Ny)                                       # predicted outputs
y = SX.sym('y', Nx)

ode = getODE(x, z, u, p, dz)                                     # Symbolic ode
alg = getALG(x, z, u, p)                                     # Symbolic alg

# Formulate system dynamics as implicit function
res_ode = xdot - ode                                        # LHS - RHS
res_alg = alg
res = vertcat(res_ode, res_alg)

ffcn = Function('ffcn', [t,xdot,x,z,u,p],[res])                          # Righthand side of ode

cost = stage_cost(yp, ym)

lcost = Function('lcost', [yp, ym], [cost])

output1 = getOutputs(x, z, u, p)

output = Function('output', [x,p], [output1])


dae = {'x':x, 'z':z, 'p':vertcat(u, p), 'ode':ode, 'alg':alg}

# Setup plant for simulation
opts = {"tf":h, "linear_solver":"csparse"} # interval length
I = integrator('I', 'idas', dae, opts)

# sol = I(x0=x_init, z0=[-30, -26], p=[irrigation[0], 1, 1, 1, 1, 1])  # Here, it is ok to not have z0.

G, G4 = ini_state(y_init, [0.9, 0.9, 0.9, 0.9, 0.9], pars)
GL = np.zeros((len(timeList), numberOfNodes))
GL[0] = G
Z = [G[0], G[-1]]

for i in range(len(timeList)-1):
    print('From', i, ' min(s), to ', i + 1, ' min(s)')
    if i == 414:
        pass
    Ik = I(x0=G, z0=Z, p=vertcat(irrigation[i], [0.9, 0.9, 0.9, 0.9, 0.9]))  # integrator with initial state G, and input U[k]
    G = Ik['xf']  # Assign the finial state to the initial state
    G_np = Ik['xf'].full().ravel()
    GL[i+1] = G_np
    Z = Ik['zf']
    # Z[0] = G[0]
thetaL = thetaFun_nofabs(GL, [0.9, 0.9, 0.9, 0.9, 0.9], pars)
theta_i = mtimes(thetaL, CMatrix.T)
difference_i = theta_i - theta_exp
obj_i = difference_i**2
obj_i_sum = sum1(obj_i)
obj_i_sum = sum2(obj_i_sum)


G, G4 = ini_state(y_init, [2.99737827e-06/pars['Ks'], 4.25461746e-01/pars['thetaS'], 6.18438096e-02/pars['thetaR'], 3.71675224e+00/pars['alpha'],
                                                 1.50828922e+00/pars['n']], pars)
GL = np.zeros((len(timeList), numberOfNodes))
GL[0] = G
Z = [G[0], G[-1]]

for i in range(len(timeList)-1):
    print('From', i, ' min(s), to ', i + 1, ' min(s)')
    Ik = I(x0=G, z0=Z, p=vertcat(irrigation[i], [2.99737827e-06/pars['Ks'], 4.25461746e-01/pars['thetaS'], 6.18438096e-02/pars['thetaR'], 3.71675224e+00/pars['alpha'],
                                                 1.50828922e+00/pars['n']]))  # integrator with initial state G, and input U[k]
    G = Ik['xf']  # Assign the finial state to the initial state
    G_np = Ik['xf'].full().ravel()
    GL[i+1] = G_np
    Z = Ik['zf']
thetaL_opt = thetaFun_nofabs(GL, [2.99737827e-06/pars['Ks'], 4.25461746e-01/pars['thetaS'], 6.18438096e-02/pars['thetaR'], 3.71675224e+00/pars['alpha'],
                                                 1.50828922e+00/pars['n']], pars)
theta_opt = mtimes(thetaL_opt, CMatrix.T)
difference_opt = theta_opt - theta_exp
obj_opt = difference_opt**2
obj_opt_sum = sum1(obj_opt)  # sum column
obj_opt_sum = sum2(obj_opt_sum)  # sum row


G, G4 = ini_state(y_init, [0.962912, 1.35, 0.45, 0.896619, 1.35], pars)
# G = array([-0.41198356,-0.6179753,-0.61797527,-0.42555029,-0.41198372,-0.41198372,-0.50193434,
#       -0.61797529,-0.61797531,-0.41198403,-0.41198357,-0.41198385,-0.61797515,-0.61797518,
#       -0.61797521,-0.61797527,-0.41198363,-0.41198383,-0.41198957,-0.61797269,-0.61797311,
#       -0.61797317,-0.6179732,-0.41210495,-0.41205005,-0.41258558,-0.61748749,-0.61741007,
#       -0.61781217,-0.61796961,-0.41198616,-0.41504543])
GL = np.zeros((len(timeList), numberOfNodes))
GL[0] = G
Z = [G[0], G[-1]]

for i in range(len(timeList)-1):
    print('From', i, ' min(s), to ', i + 1, ' min(s)')
    Ik = I(x0=G, z0=Z, p=vertcat(irrigation[i], [0.962912, 1.35, 0.45, 0.896619, 1.35]))  # integrator with initial state G, and input U[k]
    G = Ik['xf']  # Assign the finial state to the initial state
    G_np = Ik['xf'].full().ravel()
    GL[i+1] = G_np
    Z = Ik['zf']
thetaL_opt2 = thetaFun_nofabs(GL, [0.962912, 1.35, 0.45, 0.896619, 1.35], pars)
theta_opt2 = mtimes(thetaL_opt2, CMatrix.T)
difference_opt2 = theta_opt2 - theta_exp
obj_opt2 = difference_opt2**2
obj_opt_sum2 = sum1(obj_opt2)  # sum column
obj_opt_sum2 = sum2(obj_opt_sum2)  # sum row

plt.figure()
plt.plot(timeList_original/h*ratio_t, y_exp[:, 0], 'b-.', label=r'$\theta_{1, exp}$')
plt.plot(timeList/h*ratio_t, theta_i[:, 0], 'y-', label=r'$\theta_{1, ini}$')
plt.plot(timeList/h*ratio_t, theta_opt[:, 0], 'r--', label=r'$\theta_{1, est}$ shooting method')
plt.plot(timeList/h*ratio_t, theta_opt2[:, 0], 'g--', label=r'$\theta_{1, est}$ collocation method')

plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList_original/h*ratio_t, y_exp[:, 1], 'b-.', label=r'$\theta_{2, exp}$')
plt.plot(timeList/h*ratio_t, theta_i[:, 1], 'y-', label=r'$\theta_{2, ini}$')
plt.plot(timeList/h*ratio_t, theta_opt[:, 1], 'r--', label=r'$\theta_{2, est}$ shooting method')
plt.plot(timeList/h*ratio_t, theta_opt2[:, 1], 'g--', label=r'$\theta_{2, est}$ collocation method')

plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList_original/h*ratio_t, y_exp[:, 2], 'b-.', label=r'$\theta_{3, exp}$')
plt.plot(timeList/h*ratio_t, theta_i[:, 2], 'y-', label=r'$\theta_{3, ini}$')
plt.plot(timeList/h*ratio_t, theta_opt[:, 2], 'r--', label=r'$\theta_{3, est}$ shooting method')
plt.plot(timeList/h*ratio_t, theta_opt2[:, 2], 'g--', label=r'$\theta_{3, est}$ collocation method')
#
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList_original/h*ratio_t, y_exp[:, 3], 'b-.', label=r'$\theta_{4, exp}$')
plt.plot(timeList/h*ratio_t, theta_i[:, 3], 'y-', label=r'$\theta_{4, ini}$')
plt.plot(timeList/h*ratio_t, theta_opt[:, 3], 'r--', label=r'$\theta_{4, est}$ shooting method')
plt.plot(timeList/h*ratio_t, theta_opt2[:, 3], 'g--', label=r'$\theta_{4, est}$ collocation method')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

# np.savetxt('sim_results_1D_farm_robust', theta_i)
# io.savemat('sim_results_1D_farm_robust', dict(y_1D_farm=theta_e))