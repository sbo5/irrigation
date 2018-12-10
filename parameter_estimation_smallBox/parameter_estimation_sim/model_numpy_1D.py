"""
Created on Mon Oct 15 2018

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
from casadi import *
# from math import *
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
    theta = theta.ravel()
    return theta


def KFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    K = pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/(pars['m']+pars['mini'])))+pars['mini'])**pars['m']+pars['mini'])**2
    K = K.full()
    K = K.ravel()
    return K


def CFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    dSedh=pars['alpha']*pars['m']/(1-pars['m']+pars['mini'])*(Se+pars['mini'])**(1/(pars['m']+pars['mini']))*(1-(Se+pars['mini'])**(1/(pars['m']+pars['mini']))+pars['mini'])**pars['m']
    C = Se*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh
    C = C.full().ravel()
    return C


# Calculated the initial state
def ini_state_np(thetaIni, p):
    psiIni = hFun(thetaIni, p)
    # Approach: States in the same section are the same
    hMatrix = np.zeros(numberOfNodes)
    hMatrix[int(layersOfSensors[0] * nodesInPlane * ratio_z):int(layersOfSensors[1] * nodesInPlane * ratio_z)] = -5  # Top
    hMatrix[int(layersOfSensors[1] * nodesInPlane * ratio_z):int(layersOfSensors[2] * nodesInPlane * ratio_z)] = psiIni[0]
    hMatrix[int(layersOfSensors[2] * nodesInPlane * ratio_z):int(layersOfSensors[-1] * nodesInPlane * ratio_z)] = -3  # Bottom
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


def ini_state_32(thetaIni, p):
    psiIni = hFun(thetaIni, p)
    return psiIni


def psiTopFun(hTop0, p, psi, qTop, dz):
    F = psi + 0.5*dz*(-1-(qTop+max(0, hTop0/dt))/KFun(hTop0, p)) - hTop0  # Switching BC
    # F = psi + 0.5*dz*(-1-(qTop)/KFun(hTop0, p)) - hTop0  # Only unsat. BC
    # F = psi + 0.5*dz*(-1-(qTop-max(0, hTop0/dt))/KFun(hTop0, p)) - hTop0
    return F


def RichardsEQN_1D(psi, t, p, qTfun, qBot, psiTop, psiBot, qEva, qPrec, qRoot, qDrain, hAtm, hPond, dz, dt, theta0):
    psi_ini_guess = psi[0]
    psiTop = optimize.fsolve(psiTopFun, psi_ini_guess, args=(p, psi_ini_guess, qTfun, dz))  # Initial guess psiTop = psi[0]
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
    # define states and measurements arrays
    h = np.zeros(shape=(len(timeList), numberOfNodes))
    theta = np.zeros(shape=(len(timeList), numberOfNodes))
    h[0], hIni = ini_state_np(thetaIni, p)
    h0 = h[0]
    theta[0] = thetaFun(h0, p)      # Initial state of theta
    theta0 = theta[0]
    h_avg = np.zeros(shape=(len(timeList), numberOfSensors))  # 4 sensors
    h_avg[0] = hIni
    theta_avg = np.zeros(shape=(len(timeList), numberOfSensors))
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

        theta_avg[i+1] = np.matmul(CMatrix, theta0)
    return h_avg, theta_avg, theta, h


# ----------------------------------------------------------------------------------------------------------------------
# Define the geometry
# ----------------------------------------------------------------------------------------------------------------------
ratio_z = 1
lengthOfZ = 0.24  # meter
nodesInZ = int(24*ratio_z)  # Define the nodes
intervalInZ = nodesInZ  # The distance between the boundary to its adjoint states is 1/2 of delta_z

nodesInPlane = 1
numberOfNodes = nodesInZ*nodesInPlane

dz = lengthOfZ/intervalInZ

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
plt.plot(timeList/dt*ratio_t, theta_i[:, 0], 'y--', label=r'$theta_1$ initial')
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

# np.savetxt('sim_results_1D_farm_robust', theta_i)
# io.savemat('sim_results_1D_farm_robust', dict(y_1D_farm=theta_e))

