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

    hMatrix = MX.zeros(numberOfNodes)
    hMatrix[0*ratio_z:9*ratio_z] = psiIni[0]  # 1st section has 8 states
    hMatrix[9*ratio_z:16*ratio_z] = psiIni[1]  # After, each section has 7 states
    hMatrix[16*ratio_z:23*ratio_z] = psiIni[2]
    hMatrix[23*ratio_z:numberOfNodes] = psiIni[3]

    # hMatrix = MX.zeros(numberOfNodes)
    # hMatrix[int(0):int(1)] = psiIni[0]  # 1st section has 8 states
    # hMatrix[int(1):int(2)] = psiIni[1]  # After, each section has 7 states
    # hMatrix[int(2):int(3)] = psiIni[2]
    # hMatrix[int(3):numberOfNodes] = psiIni[3]

    return hMatrix, psiIni


def ini_state_np(thetaIni, p, pars):
    psiIni = hFun(thetaIni, p, pars)

    hMatrix = np.zeros(numberOfNodes)
    hMatrix[0*ratio_z:9*ratio_z] = psiIni[0]  # 1st section has 8 states
    hMatrix[9*ratio_z:16*ratio_z] = psiIni[1]  # After, each section has 7 states
    hMatrix[16*ratio_z:23*ratio_z] = psiIni[2]
    hMatrix[23*ratio_z:numberOfNodes] = psiIni[3]

    # hMatrix = MX.zeros(numberOfNodes)
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


def psiTopFun(hTop0, p, psi, qTop, dz):
    # F = psi[0] + 0.5*dz*(-1-(qTop+max(0, hTop0/h))/KFun(hTop0, pars)) - hTop0
    F = psi[0] + 0.5*dz*(-1-(qTop-max(0, hTop0/h))/KFun(hTop0, p, pars)) - hTop0
    return F


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
    # res = [xTop - (x[0]+0.5*dz*(-1-(u+SX.fmax(xTop/h, 0))/KFun(xTop, p, pars))),
    #        xBot - x[-1]]
    res = [xTop - (x[0]+0.5*dz*(-1-(u-SX.fmax(xTop/h, 0))/KFun(xTop, p, pars))),
           xBot - (x[-1]+dz/2.)]
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
# tf = 60.*335
# nk = 335
# tf = 60.*4040
# nk = 4040
tf = 60.*1000
nk = 1000
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
    # if i in range(int(1150/ratio_t), int(1216/ratio_t)):
    #     irrigation[i] = -(0.004 / (pi * r * r) / (65 * 60))
    # else:
    #     irrigation[i] = 0

    # # 336 case
    # if i in range(1, int(23/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    # elif i in range(int(61/ratio_t), int(89/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(161/ratio_t), int(189/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(249/ratio_t), int(277/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # else:
    #     irrigation[i] = 0

    # No irrigation
    irrigation[i] = 0


# ----------------------------------------------------------------------------------------------------------------------
# Parameter estimation configuration
# ----------------------------------------------------------------------------------------------------------------------
pars = Loam()
# Parameters and initial guess
p_init = np.ones(5)*1       # Initial parameter guess
p_min = p_init*0.7     # Parameter's lower bound
# p_min[1] = 0.365/pars['thetaS']  # Its a value smaller than 1
# p_min[2] = 0
p_max = p_init*1.3         # Parameter's upper bound
# p_max[1] = 10
# p_max[2] = 0.083/pars['thetaR']  # Its a value greater than 1

# y_init = array([10.1, 8.4, 8.6, 10.0])  # 336/1444/4361 case
# y_init = array([30.2, 8.8, 8.7, 10.0])  # 2683 case: left to right = top to bottom
y_init = array([36.2, 16.1, 10.1, 11.0])  # No irrigation

# Data. Size should match nk
h_exp = np.zeros((len(timeList_original), 4))  # four sensors
theta_exp = np.zeros((len(timeList_original), 4))
# with open('Data/exp_data_336.dat', 'r') as f:
with open('Data/exp_data_noirr_1001.dat', 'r') as f:
# with open('Data/exp_data_noirr_4041.dat', 'r') as f:
# with open('Data/exp_data_5L_4L_4361.dat', 'r') as f:
    wholeFile = f.readlines()
    for index, line in enumerate(wholeFile):
        oneLine = line.rstrip().split(",")
        oneLine = oneLine[1:5]
        h_temp = []
        theta_temp = []
        for index1, item in enumerate(oneLine):
            item = float(item)
            theta_temp.append(item)
            item = hFun(item, p_init, pars)
            h_temp.append(item)
        h_temp = array(h_temp, dtype='O')
        theta_temp = array(theta_temp, dtype='O')
        h_exp[index] = h_temp
        theta_exp[index] = theta_temp
y_exp = theta_exp      # Experimental measurments states/outputs
u_exp = irrigation      # Experimental inputs

# theta_exp = np.zeros((len(timeList_original), 4))
# with open('sim_results_336', 'r') as f:
#     wholeFile = f.readlines()
#     for index, line in enumerate(wholeFile):
#         oneLine = line.rstrip().split(" ")
#         oneLine = array(oneLine, dtype='O')
#         theta_exp[index] = oneLine
# y_exp = theta_exp
# u_exp = irrigation      # Experimental inputs

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

# ----------------------------------------------------------------------------------------------------------------------
# Generating the symbolic scheme
# ----------------------------------------------------------------------------------------------------------------------
Nz = 2                                                      # Number of algebraic states
Nx = 32                                                      # Number of differential states
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
opts = {"tf":h, "linear_solver":"csparse"} # interval length
I = integrator('I', 'idas', dae, opts)

# All parameter sets and irrigation amount
U = irrigation
P = MX.sym('P', Np)
G, G_ini = ini_state(y_init, P, pars)  # Initial state

# P = [1.,1.,1.,1.,1.]
# G, G_ini = ini_state_np(y_init, P, pars)  # Initial state

GL = []
GL.append(G)

GTop = G[0]
GBot = G[-1]
Z = MX.zeros(2)
Z[0] = GTop
Z[1] = GBot

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
opts = {"ipopt.linear_solver": "ma57"}
# opts = {"ipopt.linear_solver": "mumps"}

# opts["ipopt.hessian_approximation"] = 'limited-memory'
# opts["ipopt.jacobian_approximation"] = "finite-difference-values"

opts["ipopt.print_level"] = 5
opts["regularity_check"] = True
opts["verbose"] = True
# opts["ipopt.acceptable_tol"] = 1e-05;
# opts["ipopt.tol"]=1e-05
# opts['ipopt.max_iter'] = 100
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
