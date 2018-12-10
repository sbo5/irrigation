from __future__ import (absolute_import, print_function, division, unicode_literals)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
from casadi import *
from math import *
import matplotlib.pyplot as plt

start_time = time.time()
# First sections ######################################################################################################
# Define the geometry
ratio_z = 1
lengthOfZ = 0.67  # meter
nodesInZ = int(32/ratio_z)  # Define the nodes
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

# # Optimized parameters
# Ks_o = 3.32222350e-06  # [m/s]
# Theta_s_o = 3.97104031e-01
# Theta_r_o = 8.04645726e-02
# Alpha_o = 3.06000000e+00
# N_o = 1.62786282e+00

p0 = array([Ks_i, Theta_s_i, Theta_r_i, Alpha_i, N_i])
S = 0.0001  # [per m]
PET = 0.0000000070042  # [per sec]
mini = 1e-20
thetaIni = array([30.2, 8.8, 8.7, 10.0]) / 100  # 2683 case
# thetaIni = array([10.1, 8.4, 8.6, 10.0]) / 100  # 1444 case


# Calculated the initial state
def ini_state(p):
    hIni = ((((((thetaIni - p[2]) / (p[1] - p[2] + mini)) + mini) ** (
                1. / (-(1 - 1 / (p[4] + mini)) + mini)) - 1) + mini) ** (1. / (p[4] + mini))) / (-p[3] + mini)
    assignPlane = array([8.54925, 7.164, 7.164, 9.12238])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    # assignPlane = array([9, 7, 7, 9])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    section = array([nodesInPlane*assignPlane[3], nodesInPlane*(assignPlane[3]+assignPlane[2]),
                    nodesInPlane*(assignPlane[3]+assignPlane[2]+assignPlane[1]),
                    numberOfNodes])
    hMatrix = np.zeros(numberOfNodes)

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
    ks, theta_s, theta_r, alpha, n = p
    cc = 0.5*(((1+np.sign(h))*s)+
              (1-np.sign(h))*(s+((theta_s-theta_r)*alpha*n*(1-1/n))*((-1*alpha*-1*(h**2)**0.5)**(n-1))*
                              ((1+(-1*alpha*-1*(h**2)**0.5)**n)**(-(2-1/n)))))
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


# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def RichardsEQN_1D(x, t, u, p):
    # Optimized parameters
    irr = u
    state = x
    dhdt = np.zeros(numberOfNodes)


    for i in range(0, numberOfNodes):
        current_state = state[i]
        if i == 0:
            bc_zl = current_state
            bc_zu = state[i + nodesInPlane]

            # KzL = hydraulic_conductivity(bc_zl, p)
            KzL = mean_hydra_conductivity(bc_zl, state[i], p)
            KzU = mean_hydra_conductivity(state[i], bc_zu, p)
            deltaHzL = (state[i] - bc_zl) / dz
            deltaHzU = (bc_zu - state[i]) / dz
            temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
            temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
            temp4 = 0  # source term
        elif i == nodesInZ - 1:
            bc_zl = state[i - nodesInPlane]
            KzL = mean_hydra_conductivity(state[i], bc_zl, p)
            deltaHzL = (state[i] - bc_zl) / dz
            KzU1 = hydraulic_conductivity(current_state, p)
            bc_zu = current_state + dz * (-1 + irr / KzU1)
            bc_zu = sign(bc_zu)*0.0 + 0.5*(1-sign(bc_zu))*bc_zu
            # if bc_zu == 0:
            #     print('Water accumulated at t = ', t/60, 'min(s)')
            # KzU = hydraulic_conductivity(bc_zu, p)
            KzU = mean_hydra_conductivity(state[i], bc_zu, p)
            deltaHzU = (bc_zu - state[i]) / dz
            temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
            temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
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
        temp6 = temp5 / (capillary_capacity(current_state, p)+mini)
        dhdt[i] = temp6
    return dhdt


def simulate(p):
    h = zeros((interval, numberOfNodes))
    theta = zeros((interval, numberOfNodes))
    h[0], hIni = ini_state(p)
    h0 = h[0]
    # Initial state of theta
    assignPlane = array([8.54925, 7.164, 7.164, 9.12238])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    # assignPlane = array([9, 7, 7, 9])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    section = array([nodesInPlane*assignPlane[3], nodesInPlane*(assignPlane[3]+assignPlane[2]),
                    nodesInPlane*(assignPlane[3]+assignPlane[2]+assignPlane[1]),
                    numberOfNodes])
    theta[0][0: int(section[0])] = thetaIni[3]*100
    theta[0][int(section[0]):int(section[1])] = thetaIni[2]*100
    theta[0][int(section[1]):int(section[2])] = thetaIni[1]*100
    theta[0][int(section[2]):int(section[3])] = thetaIni[0]*100
    h_avg = zeros((interval, 4))  # 4 sensors
    h_avg[0] = hIni
    theta_avg = zeros((interval, 4))
    theta_avg[0] = thetaIni*100
    for i in range(interval-1):
        print('At time:', i + 1, ' min(s)')
        ts = [timeList[i], timeList[i + 1]]
        y = integrate.odeint(RichardsEQN_1D, h0, ts, args=(irr_amount[i], p))
        h0 = y[-1]
        h[i + 1] = h0

        theta0 = 100 * (
                    (p[1] - p[2]) * (1 + (-p[3] * h0) ** p[4]) ** (-(1 - 1 / (p[4]))) + p[2])
        theta[i + 1] = theta0
        theta0 = theta0[::-1]
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
        theta_avg[i + 1] = theta_avg[i + 1][::-1]
        h_avg[i + 1] = h_avg[i + 1][::-1]
        theta_avg[i + 1] = theta_avg[i + 1][::-1]
    return h_avg, theta_avg, theta, h


# Time interval
# Irrigation scheduled
ratio_t = 1
dt = 60.0/ratio_t  # second
# timeSpan = 1444  # min # 19 hour
timeSpan = 2683
# timeSpan = 4361
interval = int(timeSpan*60/dt)

timeList = []
for i in range(interval):
    current_t = i*dt
    timeList.append(current_t)
timeList = array(timeList, dtype='O')

timeList_original = []
for i in range(timeSpan):
    current_t = i*dt*ratio_t
    timeList_original.append(current_t)
timeList_original = array(timeList_original, dtype='O')

he = zeros((timeSpan, 4))  # four sensors
theta_e = zeros((timeSpan, 4))
# with open('Data/exp_data_5L_4L_4361.dat', 'r') as f:
with open('Data/exp_data_4L_2683.dat', 'r') as f:
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

irr_amount = zeros((interval, 1))
for i in range(0, interval):
    # irr_amount[i] = 0
    # # 4361 case
    # if i in range(0, 22*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    # elif i in range(59*ratio_t, 87*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(161*ratio_t, 189*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(248*ratio_t, 276*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(335*ratio_t), int(361*ratio_t)):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (25 * 60))
    # elif i in range(1590*ratio_t, 1656*ratio_t):
    #     irr_amount[i] = (0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    # else:
    #     irr_amount[i] = 0
    # 2683 case
    if i in range(1150*ratio_t, 1216*ratio_t):
        irr_amount[i] = (0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    else:
        irr_amount[i] = 0
    # # 1444 case
    # if i in range(0, 22*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    # elif i in range(59*ratio_t, 87*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(161*ratio_t, 189*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(248*ratio_t, 276*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(335*ratio_t), int(361*ratio_t)):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (25 * 60))
    # else:
    #     irr_amount[i] = 0

p_i = array([Ks_i, Theta_s_i, Theta_r_i, Alpha_i, N_i])
# p_o = array([Ks_o, Theta_s_o, Theta_r_o, Alpha_o, N_o])

h_i, theta_i, theta_i_all, h_i_all = simulate(p_i)
# h_o, theta_o, theta_i_all1, h_i_all1 = simulate(p_o)

#
# h_e = ((((((theta_e / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / p_p[4])+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
#                 -p_p[3]+mini)
# h_i = ((((((theta_i / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / (p_p[4]+mini))+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
#                 -p_p[3]+mini)
# h_o = ((((((theta_o / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / (p_p[4]+mini))+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
#                 -p_p[3]+mini)
#
plt.figure()
plt.plot(timeList/60.0, theta_e[:, 0], 'b--', label=r'$theta_1$ measured')
plt.plot(timeList/60.0, theta_i[:, 0], 'y-', label=r'$theta_1$ initial')
# plt.plot(timeList/60.0, theta_o[:, 0], 'g--', label=r'$theta_1$ 1st optimized')

plt.xlabel('Time, t (min)', fontsize = 16)
plt.ylabel('Water content (%)', fontsize = 16)
plt.legend(loc='best', fontsize = 16)
plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

# np.savetxt('sim_results_1D_farm_robust', theta_e)
# io.savemat('sim_results_1D_farm_robust', dict(y_1D_farm=theta_e))

