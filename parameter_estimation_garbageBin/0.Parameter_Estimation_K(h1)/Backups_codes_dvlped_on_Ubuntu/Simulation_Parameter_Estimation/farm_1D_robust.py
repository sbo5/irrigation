from __future__ import (absolute_import, print_function, division, unicode_literals)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
# from PyFoam.Error import error
# from PyFoam.Execution.BasicRunner import BasicRunner
# from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import mpctools as mpc
from casadi import *
from math import *
import matplotlib.pyplot as plt

start_time = time.time()
ratio_z = 1
# Define the geometry
lengthOfZ = 0.67  # meter
# Define the nodes
nodesInZ = int(32/ratio_z)
# nodesInZinOF = nodesInZ+1
nodesInPlane = 1
numberOfNodes = nodesInZ
dzList = lengthOfZ/nodesInZ
# Label the nodes
indexOfNodes = []
for i in range(0, numberOfNodes):
    indexOfNodes.append(i)
positionOfNodes = []
for k in range(0, nodesInZ):
    positionOfNodes.append([0, 0, k])

# Second sections #####################################################################################################
# Parameters
# Experimental parameters
# Loam
Ks_e = 1.04/100/3600  # [m/s]
Theta_s_e = 0.43
Theta_r_e = 0.078
Alpha_e = 0.036*100
N_e = 1.56

# # Loamy sand
# Ks_e = 14.59/100/3600  # [m/s]
# Theta_s_e = 0.41
# Theta_r_e = 0.057
# Alpha_e = 0.124*100
# N_e = 2.28

# # Sandy Loam
# Ks_e = 4.42/100/3600  # [m/s]
# Theta_s_e = 0.41
# Theta_r_e = 0.065
# Alpha_e = 0.075*100
# N_e = 1.89

# # Clay Loam
# Ks_e = 0.26/100/3600  # [m/s]
# Theta_s_e = 0.41
# Theta_r_e = 0.095
# Alpha_e = 0.019*100
# N_e = 1.31

# Optimized parameters
# Case 1.1: opt: tol = 1e-5
# Ks_p = 3.06768270e-06  # [m/s]
# Theta_s_p = 4.33767637e-01
# Theta_r_p = 8.24245169e-02
# Alpha_p = 3.64239760e+00
# N_p = 1.57132431e+00

# Case 1.2: opt: tol = 1e-5 & Max_iter = 10
# Ks_p = 3.10908681e-06  # [m/s]
# Theta_s_p = 4.34629763e-01
# Theta_r_p = 8.32471359e-02
# Alpha_p = 3.65252136e+00
# N_p = 1.57363815e+00

# Case 2
# Ks_p = 2.73968344e-06  # [m/s]
# Theta_s_p = 4.28791955e-01
# Theta_r_p = 8.29598437e-02
# Alpha_p = 3.54833444e+00
# N_p = 1.57613507e+00

# # Case 3
# Ks_p = 2.24626e-06  # [m/s]
# Theta_s_p = 4.25104179e-01
# Theta_r_p = 0.102
# Alpha_p = 3.34196165e+00
# N_p = 1.65096671e+00

# # Case 4: Add noise std = 1
# Ks_p = 1.70310472e-06  # [m/s]
# Theta_s_p = 3.82191033e-01
# Theta_r_p = 8.18208558e-02
# Alpha_p = 3.72122015e+00
# N_p = 1.49036462e+00

# # Case 5: Add noise std = 0.1
# Ks_p = 2.42345222e-06  # [m/s]
# Theta_s_p = 4.31295853e-01
# Theta_r_p = 1.03650017e-01
# Alpha_p = 3.36561545e+00
# N_p = 1.66328280e+00

# # Case: wider range
# Ks_p = 1.83129501e-06  # [m/s]
# Theta_s_p = 4.19012084e-01
# Theta_r_p = 1.14514166e-01
# Alpha_p = 3.16984028e+00
# N_p = 1.70974410e+00

# Case: wider range with noise
Ks_p = 1.62623849e-06  # [m/s]
Theta_s_p = 4.23182154e-01
Theta_r_p = 1.05490523e-01
Alpha_p = 3.10040040e+00
N_p = 1.73947925e+00

# Initial parameters
# # Case 1
# Ks_i = 0.00000288889*1.05  # [m/s]
# Theta_s_i = 0.43*1.05
# Theta_r_i = 0.078*1.05
# Alpha_i = 3.6*1.05
# N_i = 1.56*1.05

# # Case 2
# Ks_i = 0.00000288889*0.8  # [m/s]
# Theta_s_i = 0.43*1.2
# Theta_r_i = 0.078*1.2
# Alpha_i = 3.6*0.8
# N_i = 1.56*1.2

# # Case: Wider range
# Ks_i = 0.00000288889*0.7  # [m/s]
# Theta_s_i = 0.43*1.3
# Theta_r_i = 0.078*1.3
# Alpha_i = 3.6*0.7
# N_i = 1.56*1.3

# Case: Wider range with noise
Ks_i = 0.00000288889*0.7  # [m/s]
Theta_s_i = 0.43*1.3
Theta_r_i = 0.078*1.3
Alpha_i = 3.6*0.7
N_i = 1.56*1.3

S = 0.0001  # [per m]
PET = 0.0000000070042  # [per sec]
mini = 1e-100
thetaIni = array([30.0, 30.0, 30.0, 30.0]) / 100
# thetaIni = array([10.1, 8.4, 8.6, 10.0]) / 100

# Calculated the initial state
def ini_state(p):
    # Initial state
    hIni = zeros(4)
    for i in range(0, 4):
        hIni[i] = ((((((thetaIni[i] - p[2])/(p[1]-p[2]+mini))+mini)**(1./(-(1-1/(p[4]+mini))+mini)) - 1)+mini)**(1./(p[4]+mini)))/(-p[3]+mini)

    assignPlane = array([8.54925, 7.164, 7.164, 9.12238])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    section = array([nodesInPlane*assignPlane[3], nodesInPlane*(assignPlane[3]+assignPlane[2]),
                    nodesInPlane*(assignPlane[3]+assignPlane[2]+assignPlane[1]),
                    numberOfNodes])
    hMatrix = zeros(numberOfNodes)
    hMatrix[0: int(section[0])] = hIni[3]
    hMatrix[int(section[0]):int(section[1])] = hIni[2]
    hMatrix[int(section[1]):int(section[2])] = hIni[1]
    hMatrix[int(section[2]):int(section[3])] = hIni[0]
    return hMatrix, hIni


# Calculation of hydraulic conductivity
def hydraulic_conductivity(h, ks, alpha, n):
    term3 = ((1+(((-1*(alpha)*-1*(h**2+mini)**(1./2.))+mini)**(n)))+mini)
    term4 = ((term3**(-(1-1/(n+mini))))+mini)
    term5 = term4**(1./2.)
    term6 = term4**((n)/((n)-1+mini))
    term7 = (1-term6+mini)**(1-1/(n+mini))
    term8 = ((1-term7)**2)
    term1 = ((1 + sign(h)) * (ks))
    term2 = (1-sign(h))*(ks)*term5*term8
    term0 = (term1+term2)
    hc = 0.5*term0
    return hc


# Calculation of capillary capacity
def capillary_capacity(h, theta_s, theta_r, alpha, n, s=S):
    cc = 0.5*(((1+np.sign(h))*s)+
              (1-np.sign(h))*(s+((theta_s-theta_r)*(alpha)*(n)*(1-1/(n+mini)))*((-1*(alpha)*-1*((h)**2)**(0.5))**((n)-1))*
                              (((1+((-1*(alpha)*-1*((h)**2+mini)**(0.5))+mini)**(n))+mini)**(-(2-1/(n+mini))))))
    return cc


def mean_hydra_conductivity(left_boundary, right_boundary, ks, alpha, n):
    lk = hydraulic_conductivity(left_boundary, ks, alpha, n)
    rk = hydraulic_conductivity(right_boundary, ks, alpha, n)
    mk = mean([lk, rk])
    return mk


# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def ode(x, t, u, p):
    # Optimized parameters
    ks, theta_s, theta_r, alpha, n = p
    irr = u
    state = x
    # dhdt = SX.zeros(numberOfNodes)
    dhdt = zeros(numberOfNodes)
    dz = dzList
    for i in range(0, numberOfNodes):
        # print('time: ', timeList)
        # print('nodes: ', i)
        current_state = state[i]
        if i == 0:
            bc_zl = current_state
            bc_zu = state[i + nodesInPlane]
        elif i == nodesInZ - 1:
            bc_zl = state[i - nodesInPlane]
            KzU1 = hydraulic_conductivity(current_state, ks, alpha, n)
            bc_zu = current_state + dz * (-1 + irr / (KzU1+mini))
        else:
            bc_zl = state[i - nodesInPlane]
            bc_zu = state[i + nodesInPlane]

        KzL = mean_hydra_conductivity(state[i], bc_zl, ks, alpha, n)
        KzU = mean_hydra_conductivity(state[i], bc_zu, ks, alpha, n)
        deltaHzL = (state[i] - bc_zl) / dz
        deltaHzU = (bc_zu - state[i]) / dz

        temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
        temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
        temp4 = 0  # source term
        temp5 = temp2 + temp3 - temp4
        temp6 = temp5 / capillary_capacity(current_state, theta_s, theta_r, alpha, n)
        dhdt[i] = temp6
    return dhdt


def simulate(p):
    h = zeros((interval, numberOfNodes))
    h[0], hIni = ini_state(p)
    h0 = h[0]
    theta = zeros((interval, numberOfNodes))
    h_avg = zeros((interval, 4))  # 4 sensors
    h_avg[0] = hIni
    theta_avg = zeros((interval, 4))
    theta_avg[0] = thetaIni*100
    for i in range(interval-1):
        print('At time:', i+1, ' min(s)')
        ts = [timeList[i], timeList[i+1]]
        y = integrate.odeint(ode, h0, ts, args=(irr_amount[i], p))
        h0 = y[-1]
        h[i+1] = h0

        theta0 = 100 * ((p[1] - p[2]) * ((1 + (-p[3] * h0+mini) ** p[4])+mini) ** (-(1 - 1 / (p[4]+mini))) + p[2])
        theta[i+1] = theta0
        if ratio_z == 1:
            start = 2
            end = 9
            for j in range(0, 4):
                h_avg[i+1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                theta_avg[i+1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                start += 7
                end += 7
        else:
            start = 2
            end = 5
            for j in range(0, 4):
                h_avg[i+1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                theta_avg[i+1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                start += 3
                end += 3

        h_avg[i+1] = h_avg[i+1][::-1]
        theta_avg[i+1] = theta_avg[i+1][::-1]
    return h_avg, theta_avg


# Time interval
# General
# ratio_t = 1
# dt = 60.0  # second
# timeSpan = 1444  # min # 19 hour
# # timeSpan = 927
# interval = int(timeSpan*60/dt)

# Irrigation scheduled
ratio_t = 1
dt = 60.0/ratio_t  # second
timeSpan = 1444  # min # 19 hour
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

# he = zeros((timeSpan, 4))  # four sensors
# theta_e = zeros((timeSpan, 4))
# with open('Data/exp_data_5L.dat', 'r') as f:
#     whole = f.readlines()
#     for index, line in enumerate(whole):
#         one_line = line.rstrip().split(",")
#         one_line = one_line[1:5]
#         h_temp = []
#         theta_temp = []
#         for index1, item in enumerate(one_line):
#             item = float(item)
#             theta_temp.append(item)
#             temp1 = (((item/100 - Theta_r_e) / ((Theta_s_e - Theta_r_e)+mini))+mini)
#             temp2 = (temp1 ** (1. / (-(1. - (1. / (N_e+mini)))+mini)))
#             temp3 = ((temp2 - 1)+mini)
#             temp4 = (temp3 ** (1. / (N_e+mini)))
#             item = temp4 / ((-Alpha_e)+mini)
#             h_temp.append(item)
#         h_temp = array(h_temp, dtype='O')
#         theta_temp = array(theta_temp, dtype='O')
#         he[index] = h_temp
#         theta_e[index] = theta_temp

irr_amount = zeros((interval, 1))
for i in range(0, interval):
    # irr_amount[i] = 0

    # if i in range(0, 22*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    # elif i in range(59*ratio_t, 87*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(161*ratio_t, 189*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(248*ratio_t, 276*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    if i in range(int(335*ratio_t), int(361*ratio_t)):
        irr_amount[i] = (0.0001 / (pi * 0.22 * 0.22) / (25 * 60))
    else:
        irr_amount[i] = 0

p_e = [Ks_e, Theta_s_e, Theta_r_e, Alpha_e, N_e]
p_p = [Ks_p, Theta_s_p, Theta_r_p, Alpha_p, N_p]
p_i = [Ks_i, Theta_s_i, Theta_r_i, Alpha_i, N_i]
h_e, theta_e = simulate(p_e)
h_p, theta_p = simulate(p_p)
h_i, theta_i = simulate(p_i)

white_noise_e = numpy.loadtxt('sim_results_1D_farm_white_noise_widerrange', unpack=True)
white_noise_e = white_noise_e.transpose()
# white_noise_e = white_noise_e/10
theta_e += white_noise_e


h_e = ((((((theta_e / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / (p_p[4]+mini))+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
                -p_p[3]+mini)
h_p = ((((((theta_p / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / p_p[4])+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
                -p_p[3]+mini)
h_i = ((((((theta_i / 100 - p_p[2]) / ((p_p[1] - p_p[2])+mini))+mini) ** (1. / (-(1 - 1 / p_p[4])+mini)) - 1)+mini) ** (1. / (p_p[4]+mini))) / (
                -p_p[3]+mini)

# # SSE_theta_larger = sum((theta_p-theta_e)**2)
# #
# # SSE_theta_smaller = sum((theta_i-theta_e)**2)
plt.figure(1)
# plt.subplot(4, 1, 1)
plt.plot(timeList/60.0, h_e[:, 0], 'b--', label=r'$h_1$ measured')
plt.plot(timeList/60.0, h_p[:, 0], 'r:', label=r'$h_1$ optimized')
plt.plot(timeList/60.0, h_i[:, 0], 'y-', label=r'$h_1$ initial')
plt.xlabel('Time, t (min)')
plt.ylabel('Pressure head (m)')
plt.legend(loc='best')
plt.show()
#
plt.figure(5)
# # plt.subplot(4, 1, 1)
plt.plot(timeList/60.0, theta_e[:, 0], 'b--', label=r'$theta_1$ measured')
plt.plot(timeList/60.0, theta_p[:, 0], 'r:', label=r'$theta_1$ optimized')
plt.plot(timeList/60.0, theta_i[:, 0], 'y-', label=r'$theta_1$ initial')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()
#
# plt.figure(6)
# # plt.subplot(4, 1, 1)
# plt.plot(timeList/60.0, theta_e[:, 1], 'b--', label=r'$theta_2$ measured')
# plt.plot(timeList/60.0, theta_p[:, 1], 'r:', label=r'$theta_2$ optimized')
# plt.plot(timeList/60.0, theta_i[:, 1], 'y-', label=r'$theta_2$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(7)
# # plt.subplot(4, 1, 1)
# plt.plot(timeList/60.0, theta_e[:, 2], 'b--', label=r'$theta_3$ measured')
# plt.plot(timeList/60.0, theta_p[:, 2], 'r:', label=r'$theta_3$ optimized')
# plt.plot(timeList/60.0, theta_i[:, 2], 'y-', label=r'$theta_3$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(8)
# # plt.subplot(4, 1, 1)
# plt.plot(timeList/60.0, theta_e[:, 3], 'b--', label=r'$theta_4$ measured')
# plt.plot(timeList/60.0, theta_p[:, 3], 'r:', label=r'$theta_4$ optimized')
# plt.plot(timeList/60.0, theta_i[:, 3], 'y-', label=r'$theta_4$ initial')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

# np.savetxt('sim_results_1D_farm_robust', theta_e)
# io.savemat('sim_results_1D_farm_robust', dict(y_1D_farm=theta_e))

