from __future__ import (absolute_import, print_function, division, unicode_literals)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
from casadi import *
from math import *
import matplotlib.pyplot as plt

print("I am starting up")
start_time = time.time()
# First sections ######################################################################################################
# Define the geometry
ratio_z = 1
lengthOfZ = 0.67  # meter
nodesInZ = int(32 / ratio_z)  # Define the nodes
intervalInZ = nodesInZ  # The distance between the boundary to its adjoint states is 1/2 of delta_z
nodesInPlane = 1
numberOfNodes = nodesInZ * nodesInPlane
dz = lengthOfZ / intervalInZ


# Second sections #####################################################################################################
# Initial guess parameters
# # Loam
# Ks_i = 1.04/100/3600  # [m/s]
# Theta_s_i = 0.43
# Theta_r_i = 0.078
# Alpha_i = 0.036*100  # [/m]
# N_i = 1.56

# # Loamy sand
# Ks_i = 14.59/100/3600  # [m/s]
# Theta_s_i = 0.41
# Theta_r_i = 0.057
# Alpha_i = 0.124*100
# N_i = 2.28

# # Sandy Loam
# Ks_i = 4.42/100/3600  # [m/s]
# Theta_s_i = 0.41
# Theta_r_i = 0.065
# Alpha_i = 0.075*100
# N_i = 1.89

# # Clay Loam
# Ks_i = 0.26/100/3600  # [m/s]
# Theta_s_i = 0.41
# Theta_r_i = 0.095
# Alpha_i = 0.019*100
# N_i = 1.31

# # Optimized parameters
# Initial guess: loam #####################################
# Ks_p = 3.65444585e-06  # [m/s]
# Theta_s_p = 3.94909738e-01
# Theta_r_p = 7.82718194e-02
# Alpha_p = 2.75400000e+00
# N_p = 1.56184344e+00
#
# S = 0.0001  # [per m]
# PET = 0.0000000070042  # [per sec]
# p0 = array([Ks_i, Theta_s_i, Theta_r_i, Alpha_i, N_i])


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
    return pars


def bc():
    boundary = {}
# top boundary
#     boundary['qTfun'] = irrigation
    boundary['psiTop'] = []
# bottom boundary
    boundary['qBot'] = []
    boundary['psiBot'] = []
    # boundary['psiBot'] = h0[-1] + dz / 2.
# Source term
    boundary['qRoot'] = 0  #
    boundary['qDrain'] = 0
# Others
    boundary['qEva'] = 0
    boundary['qPrec'] = 0
    boundary['hPond'] = 0
    boundary['hAtm'] = -2.0804e-06
    return boundary

qBot = []
psiTop = []
psiBot = []
# psiBot = h0[-1] + dz/2.
qRoot = 0
qDrain = 0
qEva = 0
qPrec = 0
hPond = 0
hAtm = -2.0804e-06

bc = bc()
pars = Loam()
mini = 1e-20
thetaIni = array([30.2, 8.8, 8.7, 10.0])  # 2683 case: left to right = top to bottom
# thetaIni = array([10.1, 8.4, 8.6, 10.0])  # 1444 case


def hFun(theta, p, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - p[2]*pars['thetaR']) / (p[1]*pars['thetaS'] - p[2]*pars['thetaR'] + mini) + mini) ** (1. / (-(1-1/(p[4]*pars['n']+mini)) + mini))
              - 1) + mini) ** (1. / (p[4]*pars['n'] + mini))) / (-p[3]*pars['alpha'] + mini)
    return psi


def thetaFun(psi,p,pars):
    Se = if_else(psi>=0., 1., (1+(-psi*p[3]*pars['alpha']+mini)**(p[4]*pars['n'])+mini)**(-(1-1/(p[4]*pars['n']+mini))))
    theta = 100*(p[2]*pars['thetaR']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*Se)
    # theta = theta.full().ravel()
    return theta


def KFun(psi,p,pars):
    Se = if_else(psi>=0., 1., (1+(-psi*p[3]*pars['alpha']+mini)**(p[4]*pars['n'])+mini)**(-(1-1/(p[4]*pars['n']+mini))))
    K = p[0]*pars['Ks']*(Se+mini)**pars['neta']*(1-((1-(Se+mini)**(1/((1-1/(p[4]*pars['n']+mini))+mini)))+mini)**(1-1/(p[4]*pars['n']+mini))+mini)**2
    # K = K.full().ravel()
    return K


def CFun(psi,p,pars):
    Se = if_else(psi>=0., 1., (1+(-psi*p[3]*pars['alpha']+mini)**(p[4]*pars['n'])+mini)**(-(1-1/(p[4]*pars['n']+mini))))
    dSedh=p[3]*pars['alpha']*(1-1/(p[4]*pars['n']+mini))/(1-(1-1/(p[4]*pars['n']+mini))+mini)*(Se+mini)**(1/((1-1/(p[4]*pars['n']+mini))+mini))*(1-(Se+mini)**(1/((1-1/(p[4]*pars['n']+mini))+mini))+mini)**(1-1/(p[4]*pars['n']+mini))
    C = Se*pars['Ss']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*dSedh
    # C = C.full().ravel()
    return C


# Calculated the initial state
def ini_state(thetaIni, p):
    psiIni = hFun(thetaIni, p, pars)
    # Approach 1: States in the same section are the same
    hMatrix = MX.zeros(numberOfNodes)
    # hMatrix = np.zeros(numberOfNodes)
    hMatrix[0:9] = psiIni[0]  # 1st section has 8 states
    hMatrix[9:16] = psiIni[1]  # After, each section has 7 states
    hMatrix[16:23] = psiIni[2]
    hMatrix[23:numberOfNodes] = psiIni[3]

    # # Approach 2: States in the same section are the same, overlap values are averaged
    #     hMatrix = zeros(numberOfNodes)
    #     hMatrix[0:8] = psiIni[0]
    #     hMatrix[8] = mean([psiIni[0], psiIni[1]])
    #     hMatrix[9:15] = psiIni[1]
    #     hMatrix[15] = mean([psiIni[1], psiIni[2]])
    #     hMatrix[16:22] = psiIni[2]
    #     hMatrix[22] = mean([psiIni[2], psiIni[3]])
    #     hMatrix[23:numberOfNodes] = psiIni[3]

    # # Approach 3: States are interpolated
    #     x_p = [0, 5, 12, 19, 26, numberOfNodes-1]
    #     y_interp = copy(psiIni)
    #     y_p = copy(y_interp)
    #     y_p = np.insert(y_p, 0, y_interp[0]-(y_interp[1]-y_interp[0]))
    #     y_p = np.insert(y_p, 5, y_interp[-1]-(y_interp[-2]-y_interp[-1]))
    #     x = np.linspace(0, numberOfNodes-1, 32)
    #     hMatrix = interp(x, x_p, y_p)

    return hMatrix, psiIni


# Calculated the initial state
def ini_state_np(thetaIni, p):
    psiIni = hFun(thetaIni, p, pars)
    # Approach 1: States in the same section are the same
    # hMatrix = MX.zeros(numberOfNodes)
    hMatrix = np.zeros(numberOfNodes)
    hMatrix[0:9] = psiIni[0]  # 1st section has 8 states
    hMatrix[9:16] = psiIni[1]  # After, each section has 7 states
    hMatrix[16:23] = psiIni[2]
    hMatrix[23:numberOfNodes] = psiIni[3]

    # # Approach 2: States in the same section are the same, overlap values are averaged
    #     hMatrix = zeros(numberOfNodes)
    #     hMatrix[0:8] = psiIni[0]
    #     hMatrix[8] = mean([psiIni[0], psiIni[1]])
    #     hMatrix[9:15] = psiIni[1]
    #     hMatrix[15] = mean([psiIni[1], psiIni[2]])
    #     hMatrix[16:22] = psiIni[2]
    #     hMatrix[22] = mean([psiIni[2], psiIni[3]])
    #     hMatrix[23:numberOfNodes] = psiIni[3]

    # # Approach 3: States are interpolated
    #     x_p = [0, 5, 12, 19, 26, numberOfNodes-1]
    #     y_interp = copy(psiIni)
    #     y_p = copy(y_interp)
    #     y_p = np.insert(y_p, 0, y_interp[0]-(y_interp[1]-y_interp[0]))
    #     y_p = np.insert(y_p, 5, y_interp[-1]-(y_interp[-2]-y_interp[-1]))
    #     x = np.linspace(0, numberOfNodes-1, 32)
    #     hMatrix = interp(x, x_p, y_p)

    return hMatrix, psiIni


# def thetaFun1(psi,pars):
#     Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+mini)**pars['n']+mini)**(-pars['m']))
#     theta = 100*(pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se)
#     theta = theta.full().ravel()
#     return theta
#
#
# def KFun1(psi,pars):
#     Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+mini)**pars['n']+mini)**(-pars['m']))
#     K = pars['Ks']*(Se+mini)**pars['neta']*(1-((1-(Se+mini)**(1/(pars['m']+mini)))+mini)**pars['m']+mini)**2
#     K = K.full().ravel()
#     return K
#
#
# def CFun1(psi,pars):
#     Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+mini)**pars['n']+mini)**(-pars['m']))
#     dSedh=pars['alpha']*pars['m']/(1-pars['m']+mini)*(Se+mini)**(1/(pars['m']+mini))*(1-(Se+mini)**(1/(pars['m']+mini))+mini)**pars['m']
#     C = Se*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh
#     C = C.full().ravel()
#     return C
#
#
# def hToTheta(h, p):
#     ks, theta_s, theta_r, alpha, n = p
#     thetaList = 0.5*(1-sign(h))*\
#                 (100*((theta_s-theta_r)*(1+(-alpha*(-1)*((h+mini)**2+mini)**(1./2.)+mini)**n+mini)**(-(1-1/(n+mini)))+theta_r))+\
#                 0.5*(1+sign(h))*theta_s*100
#     thetaList = thetaList.full().ravel()
#     return thetaList
#
# Calculation of hydraulic conductivity
# def hydraulic_conductivity(h, p):
#     ks, theta_s, theta_r, alpha, n = p
#     term3 = (1+((-1*alpha*-1*((h+mini)**2+mini)**(1./2.)+mini)**n)+mini)
#     term4 = (term3**(-(1-1/(n+mini)))+mini)
#     term5 = term4**(1./2.)
#     term6 = term4**(n/(n-1))
#     term7 = (1-term6)**(1-1/n)
#     term8 = ((1-term7)**2)
#     term1 = ((1 + sign(h)) * ks)
#     term2 = (1-sign(h))*ks*term5*term8
#     term0 = (term1+term2)
#     hc = 0.5*term0
#     return hc
#
# # Calculation of capillary capacity
# def capillary_capacity(h, p, s=S):
#     ks, theta_s, theta_r, alpha, n = p
#     cc = 0.5*(((1+np.sign(h))*s)+
#               (1-np.sign(h))*(s+((theta_s-theta_r)*alpha*n*(1-1/n))*((-1*alpha*-1*(h**2)**0.5)**(n-1))*
#                               ((1+(-1*alpha*-1*(h**2)**0.5)**n)**(-(2-1/n)))))
#     return cc
#
# def mean_hydra_conductivity(left_boundary, right_boundary, p):
#     lk = hydraulic_conductivity(left_boundary, p)
#     rk = hydraulic_conductivity(right_boundary, p)
#     mk = mean([lk, rk])
#     return mk


# psi = np.linspace(-10, 5)
# p = np.ones(5)
# # [1., 1., 1., 1., 1.]
# pars = Loam()
# theta = thetaFun(psi, p, pars)
# C = CFun(psi, p, pars)
# K = KFun(psi, p, pars)
# theta1 = thetaFun1(psi, pars)
# C1 = CFun1(psi, pars)
# K1 = KFun1(psi, pars)
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (5.0, 10.0)
# plt.subplot(311)
# plt.plot(psi,theta)
# plt.ylabel(r'$\theta$', fontsize=20)
# plt.plot(psi, theta1)
# plt.ylabel(r'$\theta$', fontsize=20)
# plt.subplot(312)
# plt.plot(psi,C)
# plt.ylabel(r'$C$',fontsize=20)
# plt.plot(psi, C1)
# plt.ylabel(r'$C$', fontsize=20)
# plt.subplot(313)
# plt.plot(psi,K)
# plt.ylabel(r'$K$', fontsize=20)
# plt.plot(psi, K1)
# plt.ylabel(r'$K$', fontsize=20)
# plt.xlabel(r'$\psi$', fontsize=20)


# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def RichardsEQN_1D(psi, qTfun, p, qBot=qBot, psiTop=psiTop, psiBot=psiBot, qEva=qEva, qPrec=qPrec, qRoot=qRoot, qDrain=qDrain, hAtm=hAtm, hPond=hPond, dz=dz, dt=60.0):
    q = SX.zeros(numberOfNodes + 1)
    # q = np.zeros(numberOfNodes + 1)
# Lower boundary
    if qBot == []:
        if psiBot == []:
            # Free drainage: fixed flux: fixed gradient
            KBot = KFun(np.zeros(1) + psi[-1], p, pars)
            q[-1] = -KBot
            qBot = -KBot
        else:
            # Type 1 boundary: Fixed value
            KBot = KFun(np.zeros(1) + psiBot, p, pars)
            q[-1] = -KBot * ((psi[-1] - psiBot) / dz * 2 + 1.0)
            qBot = -KBot
    else:
        # Type 2 boundary
        q[-1] = qBot
        qBot = qBot

    # theta0 = SX.zeros(numberOfNodes)
    # # theta0 = np.zeros(numberOfNodes)
    # for i in range(numberOfNodes):
    #     theta0[i] = thetaFun(psi[i], p, pars)

    # Top boundary
    qIrr = qTfun
    hPond = p[-1]
    qTop = fabs(qEva) - fabs(qPrec) + qIrr - hPond / dt  # Need to update every time! Potential flux at the soil surface
    Qin = (qBot - qTop - qRoot - qDrain) * dt  # Inflow: >0, soil is saturated at the end of the time step
    Vair = sum1((p[1]*pars['thetaS'] - theta0 / 100) * abs(dz))  # [m]
    Katm = KFun(hAtm, p, pars)
    Khalf = Katm
    Emax = - Khalf * ((hAtm - psi[0]) / dz * 2. + 1.)
    # Imax = - Khalf * ((bc['hPond'] - psi[0]) / dz * 2. + 1.)  # maximum soil water flux at the soil surface
    hSur = 0
    qSur = 0
    hbc = False
    # if psi[0] >= 0:  # Soil is saturated
    #     # def sat():
    #     #     hSur = Qin
    #     #     hbc = True
    #     # def unsat():
    #     #     qSur = qTop
    #     #     hbc = False
    #     # if_else(Qin>0, sat(), unsat())
    #     if Qin > 0:  # Water in > Water out: saturation: head-based
    #         hSur = Qin
    #         hbc = True
    #     else:  # Water in < water out: unsaturated: flux-based
    #         qSur = qTop
    #         hbc = False
    # else:  # Soil is not saturated
    #     if Qin > Vair:  # There is not enough space in the soil: Saturated
    #         hSur = Qin - Vair
    #         hbc = True
    #     else:  # There is enough space in the soil: unsaturated
    #         if qTop > 0:  # Evaporation
    #             error = input('Top boundary goes to evaporation part. It is wrong')
    #             if qTop > Emax:
    #                 hSur = bc['hAtm']
    #                 hbc = True
    #             else:
    #                 qSur = qTop
    #                 hbc = False
    #         else:  # Infiltration, ponding may occur
    #             # if qTop < Imax and qTop < -p['Ks']:  # Here is saying qTop is too small
    #             ksMax = p[0]
    #             hPond = psi[0] - dz / 2 * (qTop / ksMax*pars['Ks'] + 1)
    #             if hPond >= 1.0e-06:
    #                 # print('Ponding occurs. At t = ', t/60, ' min(s)')
    #                 # hSur = 1.0e-06
    #                 hSur = 0.0013
    #                 # hSur = hPond
    #                 hbc = True
    #             else:
    #                 qSur = qTop
    #                 hbc = False
    ksMax = p[0]*pars['Ks']
    hPond = psi[0] - dz / 2 * (qTop / ksMax + 1)
    bTop = if_else(hPond >= 1.0e-06, 1.0e-06, qTop)
    # p[5] = hPond

    # if bTop == qTop:
    #     q[0] = bTop
    # else:
    #     Ksur = KFun(bTop, p, pars)
    #     q[0] = -Ksur * ((bTop - psi[0]) / dz * 2 + 1)

    q[0] = if_else(bTop == qTop, qTop, -KFun(bTop, p, pars) * ((bTop - psi[0]) / dz * 2 + 1))

    C = SX.zeros(numberOfNodes)
    Knodes = SX.zeros(numberOfNodes)
    # C = np.zeros(numberOfNodes)
    # Knodes = np.zeros(numberOfNodes)
    for i in range(numberOfNodes):
        C[i] = CFun(psi[i], p, pars)
        Knodes[i] = KFun(psi[i], p, pars)
    Kmid = SX.zeros(numberOfNodes-1)
    # Kmid = np.zeros(numberOfNodes - 1)
    for i in range(numberOfNodes-1):
        Kmid[i] = (Knodes[i] + Knodes[i + 1]) / 2

    # j = np.arange(1, numberOfNodes)
    # i = np.arange(0, numberOfNodes-1)
    # q[j] = -Kmid*((psi[i]-psi[i+1])/dz+1.)
    for j in range(1, numberOfNodes):
        q[j] = -Kmid[j-1] * ((psi[j-1] - psi[j]) / dz + 1.)

    # i = np.arange(0, numberOfNodes)
    # dhdt = (-(q[i] - q[i + 1]) / dz) / C
    dhdt = SX.zeros(numberOfNodes)
    # dhdt = np.zeros(numberOfNodes)
    for i in range(numberOfNodes):
        dhdt[i] = (-(q[i] - q[i + 1]) / dz) / C[i]
    return dhdt


def objective(x, p, theta_e):
    # Simulate objective
    # theta = SX.zeros(4)
    # for i in range(4):
    #     theta[i] = thetaFun(x[i], p, pars)

    ks = p[0]
    theta_s = p[1]
    theta_r = p[2]
    alpha = p[3]
    n = p[4]

    temp11 = ((1 + ((-(alpha*pars['alpha']) * x)+mini) ** (n*pars['n']))+mini)
    temp22 = temp11 ** (-(1 - 1. / (n*pars['n']+mini)))
    temp33 = (theta_s*pars['thetaS'] - theta_r*pars['thetaR']) * temp22
    theta = 100 * (temp33 + theta_r*pars['thetaR'])

    theta_avg = MX.zeros(4)
    start = 2
    end = 9
    for j in range(0, 4):
        if j == 0:
            theta_avg[j] = (sum1(theta[(start - 1) * nodesInPlane:end * nodesInPlane]) / (
                    (end - (start - 1)) * nodesInPlane))
        else:
            theta_avg[j] = (sum1(theta[start * nodesInPlane:end * nodesInPlane]) / (
                    (end - start) * nodesInPlane))
        start += 7
        end += 7
    obj = (theta_avg - theta_e)**2
    obj = sum1(obj)
    return obj


# Main ###############################################################################################################
# Time interval
# Irrigation scheduled
ratio_t = 1
dt = 60.0/ratio_t  # second
# timeSpan = 1444  # min # 19 hour
timeSpan = 2683
# timeSpan = 4361
interval = int(timeSpan*60/dt)

timeList = np.arange(0, interval)*dt
timeList_original = np.arange(0, timeSpan)*dt

h_e = zeros((timeSpan, 4))  # four sensors
theta_e = zeros((timeSpan, 4))
p = [1., 1., 1., 1., 1.]
# with open('Data/exp_data_5L_1444.dat', 'r') as f:
with open('Data/exp_data_4L_2683.dat', 'r') as f:
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
            item = hFun(item, p, pars)
            h_temp.append(item)
        h_temp = array(h_temp, dtype='O')
        theta_temp = array(theta_temp, dtype='O')
        h_e[index] = h_temp
        theta_e[index] = theta_temp

irrigation = np.zeros(interval)
for i in range(0, len(irrigation)):
    # # 4361 case
    # if i in range(0, 22*ratio_t):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    # elif i in range(59*ratio_t, 87*ratio_t):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(161*ratio_t, 189*ratio_t):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(248*ratio_t, 276*ratio_t):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(335*ratio_t), int(361*ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (25 * 60))
    # elif i in range(1590*ratio_t, 1656*ratio_t):
    #     irrigation[i] = -(0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    # else:
    #     irrigation[i] = 0
    # if i in range(1*ratio_t, 1216*ratio_t):
    if i in range(1150*ratio_t, 1216*ratio_t):
        irrigation[i] = -(0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    else:
        irrigation[i] = 0


Ks = 1.
ThetaS = 1.
ThetaR = 1.
Alpha = 1.
N = 1.
hPond = 0.0
p0 = array([Ks, ThetaS, ThetaR, Alpha, N, hPond])
# plb = array([Ks*0.9, 0.37/pars['thetaS'], ThetaR*0.9, Alpha*0.9, N*0.94])  # thetaS and thetaR's constraints are modified based on data, n's is modified based on trail and error
# pub = array([Ks*1.1, ThetaS*1.1, 0.083/pars['thetaR'], Alpha*1.1, N*1.1])
plb = array([Ks, 0.37/pars['thetaS'], ThetaR, Alpha, N])  # thetaS and thetaR's constraints are modified based on data, n's is modified based on trail and error
pub = array([Ks, ThetaS*1.1, ThetaR, Alpha, N])

# Create symbolic variables
x = SX.sym('x', numberOfNodes, 1)
u = SX.sym('u')
k = SX.sym('k', len(p0), 1)
# xe = SX.sym('xe', 4, 1)  # 4 sensors

# Create ODE and Objective functions
print("I am ready to create symbolic ODE")
# x, G_ini = ini_state(thetaIni, p0)
# u = irrigation[0]
# k = p0
f_x = RichardsEQN_1D(x, u, k)
# f_q = objective(x, k, xe)

# Create integrator
print("I am ready to create integrator")
ode = {'x': x, 'p': vertcat(u, k), 'ode': f_x}
opts = {'tf': 60/ratio_t, 'regularity_check': True}  # seconds
I = integrator('I', 'cvodes', ode, opts)  # Build Casadi integrator

# All parameter sets and irrigation amount
U = irrigation
K = MX.sym('K', 6)
XE = theta_e
GL = []
# Construct graph of integrator calls
print("I am ready to create construct graph")
J = 0  # Initial cost function
# for i in range(interval):
#     if i == 0:
#         G, G_ini = ini_state(thetaIni, K)  # Initial state
#     else:
#         pass
#     Ik = I(x0=G, p=vertcat(U[i], K))  # integrator with initial state G, and input U[k]
#     # if i % (ratio_t*10) == 0:
#     #     j = int(i/(ratio_t*10))
#     #     J += objective(G, K, XE[j])
#     if i % ratio_t == 0:
#         j = int(i/ratio_t)
#         J += objective(G, K, XE[j])
#     GL.append(G)
#     G = Ik['xf']  # Assign the finial state to the initial state


# This is used to check if Casadi model return the sam results as mpctool/odeint
K = p0  # Here, K needs to contain real numbers
GL_numpy_array = []
for i in range(interval):
    if i == 0:
        G, G_ini = ini_state_np(thetaIni, K)  # Initial state
    else:
        pass
    Ik = I(x0=G, p=vertcat(U[i], K))  # integrator with initial state G, and input U[k]
    if i % (ratio_t*10) == 0:  # Here, we don actually use the objective function
        j = int(i/(ratio_t*10))
        J += objective(G, K, XE[j])
    if i == 0:
        GL_numpy_array.append(G)
    else:
        G_numpy_array = G.full()  # Convert MX to numpy array
        G_numpy_array = G_numpy_array.ravel()
        GL_numpy_array.append(G_numpy_array)
    G = Ik['xf']  # Assign the finial state to the initial state
GL_numpy_array = array(GL_numpy_array, dtype='O')

# Convert h to theta
temp11 = ((1 + ((-(K[3]*pars['alpha']) * GL_numpy_array) + mini) ** (K[4]*pars['n'])) + mini)
temp22 = temp11 ** (-(1 - 1. / (K[4]*pars['n'] + mini)))
temp33 = (K[1]*pars['thetaS'] - K[2]*pars['thetaR']) * temp22
thetaL_numpy_array = 100 * (temp33 + K[2]*pars['thetaR'])

# Take the average
thetaL_numpy_array_avg = zeros((interval, 4))
for index in range(interval):
    for i in range(0, 4):
        start = 2
        end = 9
        total = 0
        for j in range(0, 7):
            k = start + i * (end - start)
            total += thetaL_numpy_array[index][k]
            start += 1
            end += 1
        theta_avg = total / (end - start)
        thetaL_numpy_array_avg[index][-(i+1)] = theta_avg
np.savetxt('sim_results_1D_farm_casadi_2683', thetaL_numpy_array_avg)



print("I am doing creating NLP solver function")
# Allocate an NLP solver
nlp = {'x': K, 'f': J, 'g': vertcat(*GL)}  # x: Solve for P (parameters), which gives the lowest J (cost fcn), with the constraints G (propagated model)
# nlp = {'x': K, 'f': J, 'g': G}  # x: Solve for P (parameters), which gives the lowest J (cost fcn), with the constraints G (propagated model)
opts = {"ipopt.linear_solver":"ma97"}
# opts = {"ipopt.linear_solver": "ma57"}
# opts = {"ipopt.linear_solver": "mumps"}
opts["ipopt.hessian_approximation"] = 'limited-memory'
opts["ipopt.print_level"] = 5
opts["regularity_check"] = True
opts["verbose"] = True
opts["ipopt.tol"]=1e-05
opts['ipopt.max_iter'] = 30
print("I am ready to build")
solver = nlpsol('solver', 'ipopt', nlp, opts)

print("I am ready to solve")

sol = solver(
        lbx=plb,
        ubx=pub,
        x0=p0  # Initial guess of decision variable
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
p0[0] = p0[0]*pars['Ks']
p0[1] = p0[1]*pars['thetaS']
p0[2] = p0[2]*pars['thetaR']
p0[3] = p0[3]*pars['alpha']
p0[4] = p0[4]*pars['n']
print ("")
print ("Actual value(s) is(are): " + str(p0))

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))
# np.savetxt('sim_results_1D_farm_white_noise', rand_e)
# # io.savemat('sim_results_1D_odeint.mat', dict(y_1D_odeint=theta_i))
