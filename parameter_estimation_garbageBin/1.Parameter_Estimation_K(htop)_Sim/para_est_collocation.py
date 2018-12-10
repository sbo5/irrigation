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
# tf = 60.*100
# nk = 100
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
c = 10                                           # Number of collocation points (start and end points included)
d = c-1                                          # Degree of interpolating polynomial
cost_type = 0                               # 0 = SSE (sum of squared error) or 1 = ASSE (Average of SSE)
use_jit = 1                                 # 0 for No, 1 for Yes. Use JIT (Just-in-time) compiler for added speed. For small problems JIT does not make a difference.
pars = Loam()

# Bounds on variables
# Parameters and initial guess
p_init = np.ones(5)*0.9       # Initial parameter guess
p_min = p_init*0.5     # Parameter's lower bound
# p_min[1] = 0.365/pars['thetaS']  # Its a value smaller than 1
# p_min[2] = 0
p_max = p_init*1.5         # Parameter's upper bound
# p_max[1] = 10
# p_max[2] = 0.083/pars['thetaR']  # Its a value greater than 1

# y_init = np.array([10.1, 8.4, 8.6, 10.0])  # 1444 case
# y_init = array([30.2, 8.8, 8.7, 10.0])  # 2683 case: left to right = top to bottom
y_init = array([30., 30., 30., 30.])

x_init, x_init4 = ini_state(y_init, p_init, pars)       # Initial state guess
xi_min = x_init*1.2      # Initial condition/state [1]
xi_max = x_init*0.8      # Initial condition/state. Same value as [1]

x_lowerbound = -inf
# Differential state bounds and initial guess
x_min = np.ones(nodesInZ)*(x_lowerbound)         # Lower bound on states
x_max = np.zeros(nodesInZ)      # Upper bound on states

# Algebraic state bounds and initial guess
z_min = np.array([x_lowerbound, x_lowerbound])           # Lower bound on states
z_max = np.array([0, 0])            # Upper bound on states
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
Ny = 4                                                      # Number of outputs
Np = 5                                                      # Number of parameters to estimate

# Declare variables (use scalar graph)
t = SX.sym("t")                                            # time

xdot = SX.sym("xdot", Nx)                                   # xdot
x = SX.sym("x", Nx)                                        # Differential state
z = SX.sym("z", Nz)                                        # Algebraic state
u = SX.sym("u", Nu)                                        # control

ym = SX.sym("ym", Ny)                                       # measured outputs
yp = SX.sym("yp", Ny)                                       # predicted outputs
y = SX.sym('y', Nx)
p = SX.sym("p", Np)                                         # parameters

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

output = Function('output', [x, p], [output1])

# ---------------------------------------------------------------------
# NLP forumation, constraints, bounds and collocation setup. Do not touch anything here unless you know what you are doing. Consult decardin@ualberta.ca
# ---------------------------------------------------------------------
tau_root = [0] + collocation_points(d, "radau")  # Choose collocation points
C = np.zeros((d + 1, d + 1))  # Coefficients of the collocation equation
D = np.zeros(d + 1)  # Coefficients of the continuity equation

# Coefficients of the quadrature function
# F = np.zeros(d+1)

# Construct polynomial basis
for j in range(d + 1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    pol = np.poly1d([1])
    for r in range(d + 1):
        if r != j:
            pol *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = pol(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(pol)
    for r in range(d + 1):
        C[j, r] = pder(tau_root[r])

    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
#  pint = np.polyint(pol)
#  F[j] = pint(1.0)

# All collocation time points
T = np.zeros((nk, d + 1))
for k in range(nk):
    for j in range(d + 1):
        T[k, j] = h * (k + tau_root[j])

# Control bounds
u_min = u_exp
u_max = u_exp
u_init = u_exp

# Total number of variables
NX = nk * (d + 1) * Nx  # Number of collocated differential states
NZ = nk * (d) * Nz  # Number of collocated algebraic states
NU = nk * Nu  # Parametrized controls
NXF = Nx  # Final state
NP = Np
NV = NX + NZ + NU + NXF + NP

# NLP variable vector
V = MX.sym("V", NV)

# All variables with bounds and initial guess
vars_lb = np.zeros(NV)
vars_ub = np.zeros(NV)
vars_init = np.zeros(NV)
offset = 0

# Get collocated states and parametrized control
X = np.resize(np.array([], dtype=MX), (nk + 1, d + 1))
Z = np.resize(np.array([], dtype=MX), (nk, d))
U = np.resize(np.array([], dtype=MX), nk)
P = np.resize(np.array([], dtype=MX), 1)

for k in range(nk):
    # Collocated states
    for j in range(d + 1):
        # Get the expression for the state vector
        X[k, j] = V[offset:offset + Nx]

        if j != 0:
            Z[k, j - 1] = V[offset + Nx:offset + Nx + Nz]

        #  # Add the initial condition and bounds
        if k == 0 and j == 0:
            vars_init[offset:offset + Nx] = x_init
            vars_lb[offset:offset + Nx] = xi_min
            vars_ub[offset:offset + Nx] = xi_max
            offset += Nx
        else:
            if j != 0:
                vars_init[offset:offset + Nx + Nz] = np.append(x_init, z_init)
                vars_lb[offset:offset + Nx + Nz] = np.append(x_min, z_min)
                vars_ub[offset:offset + Nx + Nz] = np.append(x_max, z_max)
                offset += Nx + Nz
            else:
                vars_init[offset:offset + Nx] = x_init
                vars_lb[offset:offset + Nx] = x_min
                vars_ub[offset:offset + Nx] = x_max
                offset += Nx

    # Parametrized controls
    U[k] = V[offset:offset + Nu]
    vars_lb[offset:offset + Nu] = u_min[k]
    vars_ub[offset:offset + Nu] = u_max[k]
    vars_init[offset:offset + Nu] = u_init[k]
    offset += Nu

# State at end time
X[nk, 0] = V[offset:offset + Nx]
vars_lb[offset:offset + Nx] = x_min
vars_ub[offset:offset + Nx] = x_max
vars_init[offset:offset + Nx] = x_init
offset += Nx

# Parameter
P = V[offset:offset + Np]
vars_lb[offset:offset + Np] = p_min
vars_ub[offset:offset + Np] = p_max
vars_init[offset:offset + Np] = p_init
offset += Np

assert (offset == NV)

# Constraint function for the NLP
g = []
lbg = []
ubg = []

# Objective function
J = 0

# For all finite elements
for k in range(nk):

    # For all collocation points
    for j in range(1, d + 1):

        # Get an expression for the state derivative at the collocation point
        xp_jk = 0
        for r in range(d + 1):
            xp_jk += C[r, j] * X[k, r]

        # Add collocation equations to the NLP
        fk = ffcn(T[k, j], xp_jk / 2, X[k, j], Z[k, j - 1], U[k], P)
        g.append(fk)
        lbg.append(np.zeros(Nx + Nz))  # equality constraints
        ubg.append(np.zeros(Nx + Nz))  # equality constraints

        # Add contribution to objective
    output_2 = output(X[k, 0], P)
    qk = lcost(output_2, y_exp[k])

    if cost_type == 0:
        J += qk  # Use sum not quadrature # F[j]*qk*h
    elif cost_type == 1:
        J += qk * (1.0 / nk)  # Use sum not quadrature # F[j]*qk*h
    else:
        sys.exit("Invalid cost function type. Should be 0 (SSE) or 1 (ASSE)")

    # Get an expression for the state at the end of the finite element
    xf_k = 0
    for r in range(d + 1):
        xf_k += D[r] * X[k, r]

    # Initial state estimation constraint
    if k == 0:
        output_2 = output(X[k, 0], P)
        g.append(output_2 - y_exp[k])
        lbg.append(np.zeros(Ny))
        ubg.append(np.zeros(Ny))

    # Add continuity equation to NLP
    g.append(X[k + 1, 0] - xf_k)
    lbg.append(np.zeros(Nx))
    ubg.append(np.zeros(Nx))

# Concatenate constraints
g = vertcat(*g)

# NLP
nlp = {'x': V, 'f': J, 'g': g}

## ----
## SOLVE THE NLP
## ----

# Use just-in-time compilation to speed up the evaluation
if use_jit == 1:
    if Importer.has_plugin('clang'):
        with_jit = True
        compiler = 'clang'
    elif Importer.has_plugin('shell'):
        with_jit = True
        compiler = 'shell'
    else:
        print("WARNING; running without jit. This may result in very slow evaluation times")
        with_jit = False
        compiler = ''

# Set optimization options
opts = {}
opts["expand"] = True
opts["ipopt.max_iter"] = 200
opts["ipopt.linear_solver"] = 'ma57'
# opts["ipopt.tol"] = 1E-30

# Allocate an NLP solver
solver = nlpsol("solver", "ipopt", nlp, opts)
arg = {}

# Initial condition
arg["x0"] = vars_init

# Bounds on x
arg["lbx"] = vars_lb
arg["ubx"] = vars_ub

# Bounds on g
arg["lbg"] = np.concatenate(lbg)
arg["ubg"] = np.concatenate(ubg)

# Solve the problem
res = solver(**arg)

# ---------------------------------------------------------------------
# Solution presentation and plots
# ---------------------------------------------------------------------

print
print("=======================================")
# Print the optimal cost
print("optimal cost: ", float(res["f"]))

# Retrieve the solution
v_opt = np.array(res["x"])
print("The estimated parameters are: ")
param = res["x"][-Np:]
print(param)

results = res['x'].full()
print(results[0:32])
print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))



# The results can be retrieved and plotted. I don't know the structure of your system. We can easily do this since we have both the predicted and experimental states.