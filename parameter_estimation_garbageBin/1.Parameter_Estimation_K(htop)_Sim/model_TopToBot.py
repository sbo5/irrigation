from __future__ import (absolute_import, print_function, division, unicode_literals)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
from casadi import *
# from math import *
import matplotlib.pyplot as plt


start_time = time.time()
# First sections ######################################################################################################
# Define the geometry
ratio_z = 1
lengthOfZ = 0.67  # meter
nodesInZ = int(32*ratio_z)  # Define the nodes
intervalInZ = nodesInZ  # The distance between the boundary to its adjoint states is 1/2 of delta_z
nodesInPlane = 1
numberOfNodes = nodesInZ*nodesInPlane
dz = lengthOfZ/intervalInZ


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

pars = Loam()
pars_ini = LoamIni()
pars_opt = LoamOpt()

# thetaIni = array([30.2, 8.8, 8.7, 10.0])  # 2683 case: left to right = top to bottom
# thetaIni = array([10.1, 8.4, 8.6, 10.0])  # 1444 case
thetaIni = array([30., 30., 30., 30.])


def hFun(theta, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - pars['thetaR']) / (pars['thetaS'] - pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-pars['m'] + pars['mini']))
              - 1) + pars['mini']) ** (1. / (pars['n'] + pars['mini']))) / (-pars['alpha'] + pars['mini'])
    return psi


def thetaFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    theta = 100*(pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se)
    theta = theta.full().ravel()
    return theta


def KFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    K = pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/(pars['m']+pars['mini'])))+pars['mini'])**pars['m']+pars['mini'])**2
    K = K.full().ravel()
    return K


def CFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    dSedh=pars['alpha']*pars['m']/(1-pars['m']+pars['mini'])*(Se+pars['mini'])**(1/(pars['m']+pars['mini']))*(1-(Se+pars['mini'])**(1/(pars['m']+pars['mini']))+pars['mini'])**pars['m']
    C = Se*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh
    C = C.full().ravel()
    return C


# Calculated the initial state
def ini_state(thetaIni, p):
    psiIni = hFun(thetaIni, p)
    # Approach 1: States in the same section are the same
    hMatrix = np.zeros(numberOfNodes)
    hMatrix[int(0*ratio_z):int(9*ratio_z)] = psiIni[0]  # 1st section has 8 states
    hMatrix[int(9*ratio_z):int(16*ratio_z)] = psiIni[1]  # After, each section has 7 states
    hMatrix[int(16*ratio_z):int(23*ratio_z)] = psiIni[2]
    hMatrix[int(23*ratio_z):numberOfNodes] = psiIni[3]
    # # Approach 2: States in the same section are the same, overlap values are averaged
    # hMatrix = np.zeros(numberOfNodes)
    # hMatrix[int(0):int(1)] = psiIni[0]  # 1st section has 8 states
    # hMatrix[int(1):int(2)] = psiIni[1]  # After, each section has 7 states
    # hMatrix[int(2):int(3)] = psiIni[2]
    # hMatrix[int(3):numberOfNodes] = psiIni[3]
    return hMatrix, psiIni


def ini_state_interp(thetaIni4, p):
    psiIni4 = hFun(thetaIni4, p)
    d1 = (thetaIni4[1] - thetaIni4[0])/2.  # RHS - LHS
    firstState = np.array([thetaIni4[0] - d1])
    d2 = (thetaIni4[-1] - thetaIni4[-2])*5/6  # RHS - LHS
    lastState = np.array([thetaIni4[-1] + d2])

    thetaIni5 = np.append(firstState, thetaIni4)
    thetaIni6 = np.append(thetaIni5, lastState)

    xp = [0, 4, 11, 18, 25, 31]
    x_array = np.arange(0, 32)
    thetaIni32 = np.interp(x_array, xp, thetaIni6)

    psiIni32 = hFun(thetaIni32, p)

    return psiIni32, psiIni4


def ini_state32(thetaIni, p):
    psiIni = hFun(thetaIni, p)
    return psiIni


def psiTopFun(hTop0, p, psi, qTop, dz):
    F = psi[0] + 0.5*dz*(-1-(qTop+max(0, hTop0/dt))/KFun(hTop0, p)) - hTop0
    # F = psi[0] + 0.5*dz*(-1-(qTop-max(0, hTop0/dt))/KFun(hTop0, p)) - hTop0
    return F

# def hToTheta(h, p):
#     ks, theta_s, theta_r, alpha, n = p
#     thetaList = 0.5*(1-sign(h))*\
#                 (100*((theta_s-theta_r)*(1+(-alpha*(-1)*((h+pars['mini'])**2+pars['mini'])**(1./2.)+pars['mini'])**n+pars['mini'])**(-(1-1/(n+pars['mini'])))+theta_r))+\
#                 0.5*(1+sign(h))*theta_s*100
#     thetaList = thetaList.full().ravel()
#     return thetaList
#
# Calculation of hydraulic conductivity
# def hydraulic_conductivity(h, p):
#     ks, theta_s, theta_r, alpha, n = p
#     term3 = (1+((-1*alpha*-1*((h+pars['mini'])**2+pars['mini'])**(1./2.)+pars['mini'])**n)+pars['mini'])
#     term4 = (term3**(-(1-1/(n+pars['mini'])))+pars['mini'])
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


psi = np.linspace(-10, 5, 10000)
# theta = hToTheta(psi, p0)
# C = capillary_capacity(psi, p0, s=S)
# K = hydraulic_conductivity(psi, p0)
pars = Loam()
theta1 = thetaFun(psi, pars)
C1 = CFun(psi, pars)
K1 = KFun(psi, pars)

plt.figure()
# plt.rcParams['figure.figsize'] = (5.0, 10.0)
plt.subplot(311)
# plt.plot(psi,theta)
# plt.ylabel(r'$\theta$', fontsize=20)
plt.plot(psi,theta1)
plt.ylabel(r'$\theta$', fontsize=20)
plt.subplot(312)
# plt.plot(psi,C)
# plt.ylabel(r'$C$',fontsize=20)
plt.plot(psi,C1)
plt.ylabel(r'$C$',fontsize=20)
plt.subplot(313)
# plt.plot(psi,K)
# plt.ylabel(r'$K$', fontsize=20)
plt.plot(psi,K1)
plt.ylabel(r'$K$', fontsize=20)
plt.xlabel(r'$\psi$', fontsize=20)
plt.show()

# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def RichardsEQN_1D(psi, t, p, qTfun, qBot, psiTop, psiBot, qEva, qPrec, qRoot, qDrain, hAtm, hPond, dz, dt, theta0):
    psiTop = optimize.fsolve(psiTopFun, psi[0], args=(p, psi, qTfun, dz))  # Initial guess psiTop = psi[0]
    if psiTop <= 0:
        psiTop = psiTop  # Useless. But it helps to understand
        hPond = 0
    else:
        psiTop = psiTop  # Useless. But it helps to understand
        # hPond = 1e-06
        hPond = psiTop
    q = np.zeros(numberOfNodes+1)
    # Lower boundary is updated within each time step
    if qBot == []:
        if psiBot == []:
            # Free drainage: fixed flux: fixed gradient
            KBot = KFun(np.zeros(1) + psi[-1], p)
            q[-1] = -KBot
            qBot = -KBot
        else:
            # Type 1 boundary: Fixed value
            KBot = KFun(np.zeros(1) + psiBot, p)
            q[-1] = -KBot * ((psi[-1] - psiBot) / dz * 2 + 1.0)
            qBot = -KBot * ((psi[-1] - psiBot) / dz * 2 + 1.0)
    else:
        # Type 2 boundary
        q[-1] = qBot
        qBot = qBot
    # theta0 = thetaFun(psi, p)  # theta0 shouldn't be updated, if dt cannot be updated
    # Top boundary is updated within each time step
    qIrr = qTfun
    # dt = t-tIni+pars['mini']  # Cannot figure out when dt = 0
    qTop = abs(qEva) - abs(qPrec) - abs(qIrr) - abs(hPond)/dt  # Everything should be updated everytime! Potential flux at the soil surface
    Qin = (qBot - qTop - qRoot - qDrain)*dt  # Inflow: >0, soil is saturated at the end of the time step
    Vair = sum((p['thetaS']-theta0/100)*abs(dz))  # [m]
    Kpond = KFun(hPond, p)
    Katm = KFun(hAtm, p)
    Emax = - Katm*((hAtm-psi[0])/dz*2.+1.)
    Imax = - Kpond*((hPond-psi[0])/dz*2.+1.)  # maximum soil water flux at the soil surface
    hSur = 0
    qSur = 0

    # ksMax = p['Ks']
    # # ksMax = KFun(psiTop, p)
    # psiTop = psi[0] - dz / 2 * (qTop / ksMax + 1)
    if psi[0] >= 0:  # Soil is saturated
    # if Vair == 0:
        if Qin > 0:  # Water in > Water out: saturation: head-based
            # print('Soil column is saturated: NO space. Pond height is: ', hPond, ' m')
            hSur = Qin
            hbc = True
        else:  # Water in < water out: unsaturated: flux-based
            # print('Soil column is saturated: HAVE space')
            qSur = qTop
            hbc = False
    else:  # Soil is not saturated
        if Qin > Vair:  # There is not enough space in the soil: Saturated
            # print('There is not enough space, pond height is: ', hPond, ' m')
            hSur = Qin - Vair
            hbc = True
        else:  # There is enough space in the soil: unsaturated
            if qTop > 0:  # Evaporation
                error = input('Top boundary goes to evaporation part. It is wrong')
                if qTop > Emax:
                    hSur = hAtm
                    hbc = True
                else:
                    qSur = qTop
                    hbc = False
            else:  # Infiltration, ponding may occur
                if qTop < Imax and qTop < -p['Ks']:  # Here is saying magnitude of qTop is too big
                # if hPond >= 1.0e-06:
                # if psiTop >= 1.0e-06:
                #     print('Infiltration: pond happened, pond height is: ', hPond, ' m')
                    # hSur = 1.0e-06
                    hSur = hPond
                    hbc = True
                else:
                    # print('Infiltration: unsaturated')
                    qSur = qTop
                    hbc = False
                
    if hbc == False:
        q[0] = qSur
    else:
        Ksur = KFun(hSur, p)
        q[0] = -Ksur * ((hSur - psi[0]) / dz * 2 + 1)
    Ksur = KFun(hSur, p)
    C = CFun(psi, p)
    i = np.arange(0, numberOfNodes-1)
    Knodes = KFun(psi, p)
    Kmid = (Knodes[i] + Knodes[i+1])/2
    
    j = np.arange(1, numberOfNodes)
    q[j] = -Kmid*((psi[i]-psi[i+1])/dz+1.)

    i = np.arange(0, numberOfNodes)
    dhdt = (-(q[i]-q[i+1])/dz)/C
    return dhdt


def simulate(p):
    h = np.zeros((len(timeList), numberOfNodes))
    theta = np.zeros((len(timeList), numberOfNodes))
    h[0], hIni = ini_state(thetaIni, p)
    h0 = h[0]
    theta[0] = thetaFun(h0, p)      # Initial state of theta
    theta0 = theta[0]
    h_avg = np.zeros((len(timeList), 4))  # 4 sensors
    h_avg[0] = hIni
    theta_avg = np.zeros((len(timeList), 4))
    theta_avg[0] = thetaIni

    # Boundary conditions
    qTfun = irrigation
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
    for i in range(len(timeList)-1):  # in ts, end point is timeList[i+1], which is 2682*60
        print('From', i, ' min(s), to ', i+1, ' min(s)')
        ts = [timeList[i], timeList[i + 1]]
        if i == 414:
            pass

        y = integrate.odeint(RichardsEQN_1D, h0, ts, args=(p, qTfun[i], qBot, psiTop, psiBot, qEva, qPrec, qRoot, qDrain, hAtm, hPond, dz, dt, theta0))
        h0 = y[-1]
        h[i + 1] = h0

        theta0 = thetaFun(h0, p)
        theta[i + 1] = theta0

        psiBot = []
        # psiBot = h0[-1] + dz / 2.

        start = 2*ratio_z
        end = 9*ratio_z
        for j in range(0, 4):
            if j == 0:
                h_avg[i + 1][j] = (sum(h0[int(start-1*ratio_z) * nodesInPlane:end * nodesInPlane]) / (
                        (end - (start-1*ratio_z)) * nodesInPlane))
                theta_avg[i + 1][j] = (sum(theta0[int(start-1*ratio_z) * nodesInPlane:end * nodesInPlane]) / (
                        (end - (start-1*ratio_z)) * nodesInPlane))
            else:
                h_avg[i + 1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                theta_avg[i + 1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
            start += 7*ratio_z
            end += 7*ratio_z
        # start = 0
        # end = 1
        # for j in range(0, 4):
        #     h_avg[i + 1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
        #             (end - start) * nodesInPlane))
        #     theta_avg[i + 1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
        #             (end - start) * nodesInPlane))
        #     start += 1
        #     end += 1
    return h_avg, theta_avg, theta, h


# Time interval
# Irrigation scheduled
ratio_t = 1
dt = 60.0*ratio_t  # second
timeSpan = 720
# timeSpan = 1443  # min # 19 hour
# timeSpan = 2682
# timeSpan = 4360  # 4360 intervals, 4361 data points
interval = int(timeSpan*60/dt)

timeList_original = np.arange(0, timeSpan+1)*dt/ratio_t

timeList = np.arange(0, interval+1)*dt

# h_e = np.zeros((len(timeList_original), 4))  # four sensors
# theta_e = np.zeros((len(timeList_original), 4))
# # with open('Data/exp_data_5L_1444.dat', 'r') as f:
# # with open('Data/exp_data_4L_2683.dat', 'r') as f:
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
#             item = hFun(item, pars)
#             h_temp.append(item)
#         h_temp = array(h_temp, dtype='O')
#         theta_temp = array(theta_temp, dtype='O')
#         h_e[index] = h_temp
#         theta_e[index] = theta_temp

theta_exp = np.zeros((721, 4))
with open('sim_results_1D_farm_robust', 'r') as f:
    wholeFile = f.readlines()
    for index, line in enumerate(wholeFile):
        oneLine = line.rstrip().split(" ")
        oneLine = array(oneLine, dtype='O')
        theta_exp[index] = oneLine
theta_e = theta_exp

irrigation = np.zeros(len(timeList))
irr_interval = 3
for i in range(0, len(irrigation)):  # 1st node is constant for 1st temporal element. The last node is the end of the last temporal element.
    # # 4361 case
    # if i in range(0, int(22/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    # elif i in range(int(59/ratio_t), int(87/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(161/ratio_t), int(189/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(248/ratio_t), int(276/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(335/ratio_t), int(361/ratio_t)):
    #     irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (25 * 60))
    # elif i in range(int(1590/ratio_t), int(1656/ratio_t)):
    #     irrigation[i] = -(0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    # else:
    #     irrigation[i] = 0
    # # 2683 case
    # # if i in range(int(1/ratio_t), int(1216/ratio_t)):
    # if i in range(int(1150/ratio_t), int(1216/ratio_t)):
    #     irrigation[i] = -(0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    # else:
    #     irrigation[i] = 0
    # irrigation[i] = -(0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    if i in range(180, 540):
        irrigation[i] = -0.010/86400
    else:
        irrigation[i] = 0


h_i, theta_i, theta_i_all, h_i_all = simulate(pars)
h_opt, theta_opt, theta_opt_all, h_opt_all = simulate(pars_opt)
theta_avg = thetaFun(h_i, pars_ini)


plt.figure()
plt.plot(timeList_original/dt*ratio_t, theta_e[:, 0], 'b-.', label=r'$theta_1$ measured')
plt.plot(timeList/dt*ratio_t, theta_i[:, 0], 'y--', label=r'$theta_1$ initial')
plt.plot(timeList/dt*ratio_t, theta_opt[:, 0], 'r--', label=r'$theta_1$ optimized')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList_original/dt*ratio_t, theta_e[:, 1], 'b-.', label=r'$theta_2$ measured')
plt.plot(timeList/dt*ratio_t, theta_i[:, 1], 'y--', label=r'$theta_2$ initial')
plt.plot(timeList/dt*ratio_t, theta_opt[:, 1], 'r--', label=r'$theta_2$ optimized')

plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList_original/dt*ratio_t, theta_e[:, 2], 'b-.', label=r'$theta_3$ measured')
plt.plot(timeList/dt*ratio_t, theta_i[:, 2], 'y--', label=r'$theta_3$ initial')
plt.plot(timeList/dt*ratio_t, theta_opt[:, 2], 'r--', label=r'$theta_3$ optimized')

plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList_original/dt*ratio_t, theta_e[:, 3], 'b-.', label=r'$theta_4$ measured')
plt.plot(timeList/dt*ratio_t, theta_i[:, 3], 'y--', label=r'$theta_4$ initial')
plt.plot(timeList/dt*ratio_t, theta_opt[:, 3], 'r--', label=r'$theta_4$ optimized')

plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

# np.savetxt('sim_results_1D_farm_robust', theta_i)
# io.savemat('sim_results_1D_farm_robust', dict(y_1D_farm=theta_e))

