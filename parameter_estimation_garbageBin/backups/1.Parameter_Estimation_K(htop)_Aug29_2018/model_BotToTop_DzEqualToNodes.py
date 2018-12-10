from __future__ import (absolute_import, print_function, division, unicode_literals)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array
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
nodesInZ = int(32*ratio_z)  # Define the nodes
nodesInPlane = 1
numberOfNodes = nodesInZ*nodesInPlane
dz = lengthOfZ/nodesInZ

# Second sections #####################################################################################################
# Initial guess parameters
# Loam
Ks_i = 1.04/100/3600  # [m/s]
Theta_s_i = 0.43
Theta_r_i = 0.078
Alpha_i = 0.036*100  # [/m]
N_i = 1.56

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

p0 = array([Ks_i, Theta_s_i, Theta_r_i, Alpha_i, N_i])
S = 0.00001  # [per m]
PET = 0.0000000070042  # [per sec]
mini = 1e-20
# thetaIni = array([30.2, 8.8, 8.7, 10.0]) / 100  # 2683 case
thetaIni = array([10.1, 8.4, 8.6, 10.0]) / 100  # 1444 case


# Calculated the initial state
def ini_state(p):
    hIni = ((((((thetaIni - p[2])/(p[1]-p[2]+mini))+mini)**(1./(-(1-1/(p[4]+mini)))+mini) - 1)+mini)**(1./(p[4]+mini)))/(-p[3]+mini)

    # assignPlane = array([8.54925, 7.164, 7.164, 9.12238])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    assignPlane = array([9, 7, 7, 9])*ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    section = array([nodesInPlane*assignPlane[3], nodesInPlane*(assignPlane[3]+assignPlane[2]),
                    nodesInPlane*(assignPlane[3]+assignPlane[2]+assignPlane[1]),
                    numberOfNodes])
    hMatrix = zeros(numberOfNodes)
    hMatrix[0: int(section[0])] = hIni[3]
    hMatrix[int(section[0]):int(section[1])] = hIni[2]
    hMatrix[int(section[1]):int(section[2])] = hIni[1]
    hMatrix[int(section[2]):int(section[3])] = hIni[0]
    return hMatrix, hIni


def thetaFun(h, p):
    ks, theta_s, theta_r, alpha, n = p
    theta = 0.5*(1-sign(h))*(100*((theta_s-theta_r)*(1+(-alpha*(-1)*(h**2)**(1./2.))**n)**(-(1-1/n))+theta_r))\
            +0.5*(1+sign(h))*theta_s*100
    theta = theta.full().ravel()
    return theta


# Calculation of hydraulic conductivity
def hydraulic_conductivity(h, p):
    ks, theta_s, theta_r, alpha, n = p
    term3 = (1+((-1*alpha*-1*(h**2)**(1./2.))**n))
    term4 = (term3**(-(1-1/n)))
    term5 = term4**(1./2.)
    term6 = term4**(n/(n-1))
    term7 = (1-term6)**(1-1/n)
    term8 = ((1-term7)**2)
    term1 = ((1 + sign(h)) * ks)
    term2 = (1-sign(h))*ks*term5*term8
    term0 = (term1+term2)
    hc = 0.5*term0
    return hc


# Calculation of capillary capacity
def capillary_capacity(h, p, s=S):
    cc = []
    ks, theta_s, theta_r, alpha, n = p
    # for index, item in enumerate(h):
    # # cc = 0.5*(((1+np.sign(h))*s)+
    # #           (1-np.sign(h))*(s+((theta_s-theta_r)*alpha*n*(1-1/n))*((-1*alpha*-1*(h**2)**0.5)**(n-1))*
    # #                           ((1+(-1*alpha*-1*(h**2)**0.5)**n)**(-(2-1/n)))))
    #
    #     if item >= 0:
    #         c = s
    #     else:
    #         c = (s+((theta_s-theta_r)*alpha*n*(1-1/n))*((-1*alpha*item)**(n-1))*
    #                           ((1+(-1*alpha*item)**n)**(-(2-1/n))))
    #
    #     cc.append(c)
    Se = if_else(h>=0., 1., (1+abs(h*p[3])**p[4])**(-(1-1/n)))
    dSedh=p[3]*(1-1/n)/(1-(1-1/n))*(Se)**(1/((1-1/n)))*(1-(Se)**(1/((1-1/n))))**(1-1/n)
    cc = Se*s+(p[1]-p[2])*dSedh
    cc = cc.full().ravel()
    return cc


psi = np.linspace(-10, 5)
theta = thetaFun(psi, p0)
C = capillary_capacity(psi, p0, s=S)
K = hydraulic_conductivity(psi, p0)

plt.figure()
plt.subplot(311)
plt.plot(psi,theta)
plt.ylabel(r'$\theta$', fontsize=20)
plt.subplot(312)
plt.plot(psi,C)
plt.ylabel(r'$C$',fontsize=20)
plt.subplot(313)
plt.plot(psi,K)
plt.ylabel(r'$K$', fontsize=20)
plt.xlabel(r'$\psi$', fontsize=20)


def mean_hydra_conductivity(left_boundary, right_boundary, p):
    lk = hydraulic_conductivity(left_boundary, p)
    rk = hydraulic_conductivity(right_boundary, p)
    mk = mean([lk, rk])
    return mk


def psiTopFun(hTop0, p, psi, qTop, dz):
    F = psi[-1] - 0.5*dz*((qTop+max(0, hTop0))/hydraulic_conductivity(hTop0, p)+1) - hTop0
    return F


# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def RichardsEQN_1D(x, t, u, p):
    # Optimized parameters
    irr = u
    state = x
    dhdt = zeros(numberOfNodes)
    for i in range(0, numberOfNodes):
        current_state = state[i]
        if i == 0:
            bc_zl = current_state
            bc_zu = state[i + nodesInPlane]

            KzL = hydraulic_conductivity(bc_zl, p)
            # KzL = mean_hydra_conductivity(bc_zl, state[i], p)
            KzU = mean_hydra_conductivity(state[i], bc_zu, p)
            deltaHzL = (state[i] - bc_zl) / dz *2
            deltaHzU = (bc_zu - state[i]) / dz
            temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
            temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
            temp4 = 0  # source term
        elif i == nodesInZ - 1:
            bc_zl = state[i - nodesInPlane]
            KzL = mean_hydra_conductivity(state[i], bc_zl, p)
            deltaHzL = (state[i] - bc_zl) / dz

            bc_zu = optimize.fsolve(psiTopFun, state[-1], args=(p, state, -irr, dz))  # Initial guess psiTop = psi[0]
            if bc_zu <= 0:
                bc_zu = bc_zu  # Useless. But it helps to understand
                hPond = 0
                qTop = - abs(irr) - abs(hPond) / dt  # Everything should be updated everytime! Potential flux at the soil surface
                temp2 = 1 / (0.5 * 2 * dz) * (-qTop - KzL * (deltaHzL+1))  # In this code, q is positive, so its -qTop, not qTop
            else:
                print('Pond happend: height of pond is ', bc_zu, ' m')
                bc_zu = bc_zu  # Useless. But it helps to understand
                # hPond = 1e-06
                hPond = bc_zu
                KzU = hydraulic_conductivity(hPond, p)
                deltaHzU = (bc_zu - state[i]) / dz *2
                temp2 = 1 / (0.5 * 2 * dz) * (KzU * (deltaHzU+1) - KzL * (deltaHzL+1))

            # KzU1 = p[0]
            # # KzU1 = hydraulic_conductivity(current_state, p)
            # bc_zu = current_state + dz * (-1 + irr / KzU1)
            # # if bc_zu >= 1.0e-06:
            # #     bc_zu = 0.0013
            # #     # print('Water accumulated at t = ', t/60, 'min(s)')
            # #     # KzU = mean_hydra_conductivity(state[i], bc_zu, p)
            # #     KzU = hydraulic_conductivity(bc_zu, p)
            # #     deltaHzU = (bc_zu - state[i]) / dz
            # #     temp2 = 1 / (0.5 * 2 * dz) * (KzU * (deltaHzU+1) - KzL * (deltaHzL+1))
            # # else:
            # #     temp2 = 1 / (0.5 * 2 * dz) * (irr - KzL * (deltaHzL+1))
            # bc_zu = if_else(bc_zu>=1.0e-06, 0.0013, irr)
            # temp2 = if_else(bc_zu==irr, 1 / (0.5 * 2 * dz) * (irr - KzL * (deltaHzL+1)), 1 / (0.5 * 2 * dz) * (hydraulic_conductivity(bc_zu, p) * ((bc_zu - state[i]) / dz+1) - KzL * (deltaHzL+1)))
            temp3 = 0
            temp4 = 0  # source term
        else:
            bc_zl = state[i - nodesInPlane]
            bc_zu = state[i + nodesInPlane]

            KzL = mean_hydra_conductivity(state[i], bc_zl, p)
            KzU = mean_hydra_conductivity(state[i], bc_zu, p)
            deltaHzL = (state[i] - bc_zl) / dz
            deltaHzU = (bc_zu - state[i]) / dz
            temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
            temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
            temp4 = 0  # source term

        temp5 = temp2 + temp3 - temp4
        C = capillary_capacity(current_state, p)
        temp6 = temp5 / C
        dhdt[i] = temp6
    return dhdt


def simulate(p):
    h = zeros((len(timeList), numberOfNodes))
    theta = zeros((len(timeList), numberOfNodes))
    h[0], hIni = ini_state(p)
    h0 = h[0]
    # Initial state of theta
    # assignPlane = array([8.54925, 7.164, 7.164, 9.12238])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    assignPlane = array([9, 7, 7, 9])*ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    section = array([nodesInPlane*assignPlane[3], nodesInPlane*(assignPlane[3]+assignPlane[2]),
                    nodesInPlane*(assignPlane[3]+assignPlane[2]+assignPlane[1]),
                    numberOfNodes])
    theta[0][0: int(section[0])] = thetaIni[3]*100
    theta[0][int(section[0]):int(section[1])] = thetaIni[2]*100
    theta[0][int(section[1]):int(section[2])] = thetaIni[1]*100
    theta[0][int(section[2]):int(section[3])] = thetaIni[0]*100
    h_avg = zeros((len(timeList), 4))  # 4 sensors
    h_avg[0] = hIni
    theta_avg = zeros((len(timeList), 4))
    theta_avg[0] = thetaIni*100
    # h0 = array([-1.35221879e+01, -1.35220527e+01, -2.51122027e+01, -2.51122533e+01,
    #             -2.51114944e+01, -2.51117908e+01, -2.39660154e+01, -1.44426219e+01,
    #             -1.35220317e+01, -7.73809897e+01, -7.73809913e+01, -7.73809915e+01,
    #             -7.73809915e+01, -7.73809916e+01, -7.73809863e+01, -7.73809950e+01,
    #             -1.14548739e+02, -1.14548772e+02, -1.25454050e+02, -2.12733603e+02,
    #             -2.12733585e+02, -2.12733577e+02, -2.12733551e+02, -7.15600545e-01,
    #             -7.15600471e-01, -7.15600325e-01, -7.15600053e-01, -7.10113639e-01,
    #             -3.85321880e-01, -3.85314958e-01, -3.08791171e-01, -2.07491109e-01])
    # theta0 = thetaFun(h0, p)
    # start = 2
    # end = 9
    # for j in range(0, 4):
    #     h_avg[0][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
    #             (end - start) * nodesInPlane))
    #     theta_avg[0][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
    #             (end - start) * nodesInPlane))
    #     start += 7
    #     end += 7
    #     h_avg[0] = h_avg[0][::-1]
    #     theta_avg[0] = theta_avg[0][::-1]
    for i in range(len(timeList)-1):
        print('At time:', i + 1, ' min(s)')
        if i==177:
            pass
        ts = [timeList[i], timeList[i + 1]]
        y = integrate.odeint(RichardsEQN_1D, h0, ts, args=(irr_amount[i], p))
        h0 = y[-1]
        h[i + 1] = h0

        theta0 = 100 * (
                    (p[1] - p[2]) * (1 + (-p[3] * h0) ** p[4]) ** (-(1 - 1 / (p[4]))) + p[2])
        theta[i + 1] = theta0
        theta0 = theta0[::-1]

        start = 2*ratio_z
        end = 9*ratio_z
        for j in range(0, 4):
            if j == 0:
                h_avg[i + 1][j] = (sum(h0[(start-1*ratio_z) * nodesInPlane:end * nodesInPlane]) / (
                        (end - (start-1*ratio_z)) * nodesInPlane))
                theta_avg[i + 1][j] = (sum(theta0[(start-1*ratio_z) * nodesInPlane:end * nodesInPlane]) / (
                        (end - (start-1*ratio_z)) * nodesInPlane))
            else:
                h_avg[i + 1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                theta_avg[i + 1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
            start += 7*ratio_z
            end += 7*ratio_z
        theta_avg[i + 1] = theta_avg[i + 1][::-1]
        h_avg[i + 1] = h_avg[i + 1][::-1]
        theta_avg[i + 1] = theta_avg[i + 1][::-1]
    return h_avg, theta_avg, theta, h


# Time interval
# Irrigation scheduled
ratio_t = 1
dt = 60.0*ratio_t  # second
# timeSpan = 1443  # min # 19 hour
# timeSpan = 2682
timeSpan = 4360
interval = int(timeSpan*60/dt)

timeList = []
for i in range(interval+1):
    current_t = i*dt
    timeList.append(current_t)
timeList = array(timeList, dtype='O')

timeList_original = []
for i in range(timeSpan+1):
    current_t = i*dt/ratio_t
    timeList_original.append(current_t)
timeList_original = array(timeList_original, dtype='O')

he = zeros((len(timeList_original), 4))  # four sensors
theta_e = zeros((len(timeList_original), 4))
with open('Data/exp_data_5L_4L_4361.dat', 'r') as f:
# with open('Data/exp_data_4L_2683.dat', 'r') as f:
# with open('Data/exp_data_5L_1444.dat', 'r') as f:
    whole = f.readlines()
    for index, line in enumerate(whole):
        one_line = line.rstrip().split(",")
        one_line = one_line[1:5]
        h_temp = []
        theta_temp = []
        for index1, item in enumerate(one_line):
            item = float(item)
            theta_temp.append(item)
            temp1 = (((item/100 - Theta_r_i) / ((Theta_s_i - Theta_r_i))))
            temp2 = (temp1 ** (1. / (-(1. - (1. / (N_i))))))
            temp3 = ((temp2 - 1))
            temp4 = (temp3 ** (1. / (N_i)))
            item = temp4 / ((-Alpha_i))
            h_temp.append(item)
        h_temp = array(h_temp, dtype='O')
        theta_temp = array(theta_temp, dtype='O')
        he[index] = h_temp
        theta_e[index] = theta_temp

irr_amount = zeros((len(timeList), 1))
for i in range(0, len(timeList)):
    # irr_amount[i] = 0
    # 4361 case
    if i in range(0, int(22/ratio_t)):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    elif i in range(int(59/ratio_t), int(87/ratio_t)):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    elif i in range(int(161/ratio_t), int(189/ratio_t)):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    elif i in range(int(248/ratio_t), int(276/ratio_t)):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    elif i in range(int(335/ratio_t), int(361/ratio_t)):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (25 * 60))
    elif i in range(int(1590/ratio_t), int(1656/ratio_t)):
        irr_amount[i] = (0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    else:
        irr_amount[i] = 0
    # if i in range(1150*ratio_t, 1216*ratio_t):
    #     irr_amount[i] = (0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    # else:
    #     irr_amount[i] = 0

p_i = array([Ks_i, Theta_s_i, Theta_r_i, Alpha_i, N_i])
# p_i1 = array([Ks_i1, Theta_s_i1, Theta_r_i1, Alpha_i1, N_i1])
# p_i2 = array([Ks_i2, Theta_s_i2, Theta_r_i2, Alpha_i2, N_i2])
# p_p = array([Ks_p, Theta_s_p, Theta_r_p, Alpha_p, N_p])
h_i, theta_i, theta_i_all, h_i_all = simulate(p_i)
# h_i1, theta_i1, theta_i_all1, h_i_all1 = simulate(p_i1)
# h_i2, theta_i2, theta_i_all2, h_i_all2 = simulate(p_i2)
# h_p, theta_p, theta_p_all, h_p_all = simulate(p_p)
#
# h_e = ((((((theta_e / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / p_p[4])+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
#                 -p_p[3]+mini)
# h_i = ((((((theta_i / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / (p_p[4]+mini))+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
#                 -p_p[3]+mini)
# h_p = ((((((theta_p / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / (p_p[4]+mini))+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
#                 -p_p[3]+mini)
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, -1], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, 1], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, 2], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, 3], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, 4], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, 5], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, 6], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, 7], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# # plt.rcParams['figure.figsize'] = (10.0, 5.0)
#
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_i_all[:, 8], 'y-', label=r'$theta_1$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()


plt.figure()
plt.plot(timeList_original/dt*ratio_t, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
plt.plot(timeList/dt*ratio_t, theta_i[:, 0], 'y-', label=r'$theta_1$ initial')
# plt.plot(timeList/60.0, theta_i1[:, 0], 'g--', label=r'$theta_1$ 1st optimized')
# plt.plot(timeList/60.0, theta_i2[:, 0], 'r--', label=r'$theta_1$ 2nd optimized')
# plt.plot(timeList/60.0, theta_p[:, 0], 'b--', label=r'$theta_1$ final optimized')
plt.xlabel('Time, t (min)', fontsize = 16)
plt.ylabel('Water content (%)', fontsize = 16)
plt.legend(loc='best', fontsize = 16)
plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

# np.savetxt('sim_results_1D_farm_robust', theta_e)
# io.savemat('sim_results_1D_farm_robust', dict(y_1D_farm=theta_e))

