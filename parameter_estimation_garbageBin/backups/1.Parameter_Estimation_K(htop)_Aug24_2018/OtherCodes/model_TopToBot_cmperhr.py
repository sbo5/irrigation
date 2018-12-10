from __future__ import (absolute_import, print_function, division, unicode_literals)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
import mpctools as mpc
from casadi import *
from math import *
import matplotlib.pyplot as plt

start_time = time.time()
# First sections ######################################################################################################
# Define the geometry
ratio_z = 1
lengthOfZ = 0.67  # meter
nodesInZ = int(32/ratio_z)  # Define the nodes
intervalInZ = nodesInZ  # The distance between the boundary to its adjoint states is 1/2 of delta_z
nodesInPlane = 1
numberOfNodes = nodesInZ*nodesInPlane
dz = lengthOfZ/intervalInZ

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
    pars['alpha'] = 0.036  #*100  Unit changed
    pars['n'] = 1.56
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 1.04  #/100/3600  Unit changed
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
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
    return pars


pars = Loam()
mini = 1e-20
# thetaIni = array([30.2, 8.8, 8.7, 10.0])  # 2683 case: left to right = top to bottom
thetaIni = array([10.1, 8.4, 8.6, 10.0])  # 1444 case
# thetaIni = array([10, 30.0, 30.0, 30.0])  # 1444 case


def hFun(theta, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - pars['thetaR']) / (pars['thetaS'] - pars['thetaR'] + mini) + mini) ** (1. / (-pars['m'] + mini))
              - 1) + mini) ** (1. / (pars['n'] + mini))) / (-pars['alpha'] + mini)
    return psi


def thetaFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+mini)**pars['n']+mini)**(-pars['m']))
    theta = 100*(pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se)
    theta = theta.full().ravel()
    return theta


def KFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+mini)**pars['n']+mini)**(-pars['m']))
    K = pars['Ks']*(Se+mini)**pars['neta']*(1-((1-(Se+mini)**(1/(pars['m']+mini)))+mini)**pars['m']+mini)**2
    K = K.full().ravel()
    return K


def CFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+mini)**pars['n']+mini)**(-pars['m']))
    dSedh=pars['alpha']*pars['m']/(1-pars['m']+mini)*(Se+mini)**(1/(pars['m']+mini))*(1-(Se+mini)**(1/(pars['m']+mini))+mini)**pars['m']
    C = Se*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh
    C = C.full().ravel()
    return C


# Calculated the initial state
def ini_state(thetaIni, p):
    psiIni = hFun(thetaIni, pars)
# Approach 1: States in the same section are the same
    hMatrix = zeros(numberOfNodes)
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


def psiTopFun(hTop0, p, psi, qTop, dz):
    F = psi[0] - 0.5*dz*(qTop/KFun(hTop0, p)+1) - hTop0
    return F

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


# psi = np.linspace(-10, 1, 10000)
# # theta = hToTheta(psi, p0)
# # C = capillary_capacity(psi, p0, s=S)
# # K = hydraulic_conductivity(psi, p0)
# pars = Loam()
# theta1 = thetaFun(psi, pars)
# C1 = CFun(psi, pars)
# K1 = KFun(psi, pars)
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (5.0, 10.0)
# plt.subplot(311)
# # plt.plot(psi,theta)
# # plt.ylabel(r'$\theta$', fontsize=20)
# plt.plot(psi,theta1)
# plt.ylabel(r'$\theta$', fontsize=20)
# plt.subplot(312)
# # plt.plot(psi,C)
# # plt.ylabel(r'$C$',fontsize=20)
# plt.plot(psi,C1)
# plt.ylabel(r'$C$',fontsize=20)
# plt.subplot(313)
# # plt.plot(psi,K)
# # plt.ylabel(r'$K$', fontsize=20)
# plt.plot(psi,K1)
# plt.ylabel(r'$K$', fontsize=20)
# plt.xlabel(r'$\psi$', fontsize=20)

# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def RichardsEQN_1D(psi, t, p, qTfun, qBot, psiTop, psiBot, qEva, qPrec, qRoot, qDrain, hAtm, hPond, dz, dt, theta0):
    q = np.zeros(numberOfNodes+1)
    # Lower boundary
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
    theta0=thetaFun(psi, p)
    # Top boundary
    qIrr = qTfun
    hPond = np.loadtxt('hPondFile0')
    hPond = float(hPond)
    qTop = abs(qEva) - abs(qPrec) - abs(qIrr) - abs(hPond)/dt  # Everything should be updated everytime! Potential flux at the soil surface
    # Iteratively calculate psiTop
    psiTop = optimize.fsolve(psiTopFun, psi[0], args=(p, psi, qTop, dz))
    Qin = (qBot - qTop - qRoot - qDrain)*dt  # Inflow: >0, soil is saturated at the end of the time step
    Vair = sum((p['thetaS']-theta0/100)*abs(dz))  # [m]
    Kpond = KFun(hPond, p)
    Khalf = Kpond
    Emax = - Khalf*((hAtm-psi[0])/dz*2.+1.)
    Imax = - Khalf*((hPond-psi[0])/dz*2.+1.)  # maximum soil water flux at the soil surface
    hSur = 0
    qSur = 0
    hbc = False

    # # ksMax = p['Ks']
    # ksMax = KFun(psiTop, p)
    # psiTop = psi[0] - dz / 2 * (qTop / ksMax + 1)
    if psi[0] >= 0:  # Soil is saturated
    # if Vair == 0:
        if Qin > 0:  # Water in > Water out: saturation: head-based
            print('First state is saturated: NO space')
            hSur = Qin
            hPond = np.array([hSur])
            np.savetxt('hPondFile0', hPond)
            hbc = True
        else:  # Water in < water out: unsaturated: flux-based
            print('First state is saturated: HAVE space')
            qSur = qTop
            hPond = np.array([0])
            np.savetxt('hPondFile0', hPond)
            hbc = False
    else:  # Soil is not saturated
        if Qin > Vair:  # There is not enough space in the soil: Saturated
            print('There is not enough space')
            hSur = Qin - Vair
            hPond = np.array([hSur])
            np.savetxt('hPondFile0', hPond)
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
                if qTop < Imax and qTop < -p['Ks']:  # Here is saying qTop is too small
                # if hPond >= 1.0e-06:
                # if psiTop >= 1.0e-06:
                    print('Infiltration: ponding happened')
                    # print('Ponding occurs. At t = ', t/60, ' min(s)')
                    # hSur = 1.0e-06
                    # hSur = 0.0013
                    # hSur = hPond
                    hSur = psiTop
                    hPond = np.array([hSur])
                    np.savetxt('hPondFile0', hPond)
                    # hbc = True
                    hbc = True
                else:
                    print('Infiltration: unsaturated')
                    qSur = qTop
                    hPond = np.array([0])
                    np.savetxt('hPondFile0', hPond)
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

    info = np.array([t, dhdt[0], C[0], qTop, q[1], Ksur, Kmid[0], hSur, psi[0], psi[1]])

    return dhdt


def simulate(p):
    h = zeros((interval, numberOfNodes))
    theta = zeros((interval, numberOfNodes))
    h[0], hIni = ini_state(thetaIni, p)
    h0 = h[0]
    theta[0] = thetaFun(h0, p)      # Initial state of theta
    theta0 = theta[0]
    h_avg = zeros((interval, 4))  # 4 sensors
    h_avg[0] = hIni
    theta_avg = zeros((interval, 4))
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
    hPond = np.zeros(1)
    np.savetxt('hPondFile0', hPond)
    hAtm = -2.0804e-06
    for i in range(interval-1):  # in ts, end point is timeList[i+1], which is 2682*60
        print('At time:', i + 1, ' min(s)')
        ts = [timeList[i], timeList[i + 1]]
        if i == 356:
            y = integrate.odeint(RichardsEQN_1D, h0, ts, args=(
            p, qTfun[i], qBot, psiTop, psiBot, qEva, qPrec, qRoot, qDrain, hAtm, hPond, dz, dt, theta0))

        y = integrate.odeint(RichardsEQN_1D, h0, ts, args=(p, qTfun[i], qBot, psiTop, psiBot, qEva, qPrec, qRoot, qDrain, hAtm, hPond, dz, dt, theta0))
        h0 = y[-1]
        h[i + 1] = h0

        theta0 = thetaFun(h0, p)
        theta[i + 1] = theta0

        psiBot = []
        # psiBot = h0[-1] + dz / 2.

        if ratio_z == 1:
            start = 2
            end = 9
            for j in range(0, 4):
                if j == 0:
                    h_avg[i + 1][j] = (sum(h0[(start-1) * nodesInPlane:end * nodesInPlane]) / (
                            (end - (start-1)) * nodesInPlane))
                    theta_avg[i + 1][j] = (sum(theta0[(start-1) * nodesInPlane:end * nodesInPlane]) / (
                            (end - (start-1)) * nodesInPlane))
                else:
                    h_avg[i + 1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                            (end - start) * nodesInPlane))
                    theta_avg[i + 1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                            (end - start) * nodesInPlane))
                start += 7
                end += 7
        else:
            start = 2
            end = 9
            for j in range(0, 4):
                h_avg[i + 1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                theta_avg[i + 1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                start += 3
                end += 3
        # h_avg[i + 1] = h_avg[i + 1][::-1]
        # theta_avg[i + 1] = theta_avg[i + 1][::-1]
    return h_avg, theta_avg, theta, h


# Time interval
# Irrigation scheduled
ratio_t = 1
dt = 60.0/ratio_t  # second
# timeSpan = 1444  # min # 19 hour
# timeSpan = 2683
timeSpan = 4361
interval = int(timeSpan*60/dt)

timeList = np.arange(0, interval)*dt

timeList_original = np.arange(0, timeSpan)*dt*ratio_t

h_e = zeros((timeSpan, 4))  # four sensors
theta_e = zeros((timeSpan, 4))
# with open('Data/exp_data_5L_1444.dat', 'r') as f:
# with open('Data/exp_data_4L_2683.dat', 'r') as f:
with open('Data/exp_data_5L_4L_4361.dat', 'r') as f:
    wholeFile = f.readlines()
    for index, line in enumerate(wholeFile):
        oneLine = line.rstrip().split(",")
        oneLine = oneLine[1:5]
        h_temp = []
        theta_temp = []
        for index1, item in enumerate(oneLine):
            item = float(item)
            theta_temp.append(item)
            item = hFun(item, pars)
            h_temp.append(item)
        h_temp = array(h_temp, dtype='O')
        theta_temp = array(theta_temp, dtype='O')
        h_e[index] = h_temp
        theta_e[index] = theta_temp

irrigation = np.zeros(interval)
# irrigation[0: 22*ratio_t] = -(0.001 / (pi * 0.22 * 0.22) / (22 * 60))
# irrigation[59*ratio_t: 87*ratio_t] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
# irrigation[161*ratio_t: 189*ratio_t] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
# irrigation[248*ratio_t: 276*ratio_t] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
# irrigation[335*ratio_t: 361*ratio_t] = -(0.001 / (pi * 0.22 * 0.22) / (25 * 60))
# irrigation[1590*ratio_t: 1656*ratio_t] = -(0.001 / (pi * 0.22 * 0.22) / (65 * 60))

irr_interval = 2
for i in range(0, len(irrigation)):
    # 4361 case
    if i in range(0, 22*ratio_t):
        irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    elif i in range(59*ratio_t, 87*ratio_t):
        irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    elif i in range(161*ratio_t, 189*ratio_t):
        irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    elif i in range(248*ratio_t, 276*ratio_t):
        irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    elif i in range(int(335*ratio_t), int(361*ratio_t)):
        irrigation[i] = -(0.001 / (pi * 0.22 * 0.22) / (25 * 60))
    elif i in range(1590*ratio_t, 1656*ratio_t):
        irrigation[i] = -(0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    else:
        irrigation[i] = 0

    # # if i in range(1*ratio_t, 1216*ratio_t):
    # # if i in range(1150*ratio_t, 1216*ratio_t):
    # #     irrigation[i] = -(0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    # # else:
    # #     irrigation[i] = 0
    # irrigation[i] = -(0.004 / (pi * 0.22 * 0.22) / (65 * 60))


h_i, theta_i, theta_i_all, h_i_all = simulate(pars)


plt.figure()
plt.plot(timeList/dt/ratio_t, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
plt.plot(timeList//dt/ratio_t, theta_i[:, 0], 'y-', label=r'$theta_1$ initial')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList//dt/ratio_t, theta_e[:, 1], 'b:', label=r'$theta_2$ measured')
plt.plot(timeList//dt/ratio_t, theta_i[:, 1], 'y-', label=r'$theta_2$ initial')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList//dt/ratio_t, theta_e[:, 2], 'b:', label=r'$theta_3$ measured')
plt.plot(timeList//dt/ratio_t, theta_i[:, 2], 'y-', label=r'$theta_3$ initial')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(timeList//dt/ratio_t, theta_e[:, 3], 'b:', label=r'$theta_4$ measured')
plt.plot(timeList//dt/ratio_t, theta_i[:, 3], 'y-', label=r'$theta_4$ initial')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

# np.savetxt('sim_results_1D_farm_robust', theta_e)
# io.savemat('sim_results_1D_farm_robust', dict(y_1D_farm=theta_e))

