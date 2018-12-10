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

# Define the geometry
lengthOfZ = 0.67  # meter
# Define the nodes
nodesInZ = 32
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
# Initial parameters
Ks = 0.00000288889  # [m/s]
Theta_s = 0.43
Theta_r = 0.078
Alpha = 3.6
N = 1.56

# Predicted parameters
Ksp = 2.12332461e-06  # [m/s]
Theta_sp = 4.67994846e-01
Theta_rp = 8.40000000e-02
Alphap = 3.99999969e+00
Np = 1.58817886e+00

S = 0.0001  # [per m]
PET = 0.0000000070042  # [per sec]
mini= 1e-100
# Initial state
# thetaIni = array([26.8, 10.7, 9.0, 10.3])/100
thetaIni = array([10.1, 8.4, 8.6, 10.0])/100
hIni = zeros(4)
for i in range(0, 4):
    hIni[i] = ((((thetaIni[i] - Theta_r)/(Theta_s-Theta_r))**(1./(-(1-1/N))) - 1)**(1./N))/(-Alpha)

assignPlane = array([8.54925, 7.164, 7.164, 9.12238])  # the sum of assignPlane need to be the same as nodesInZ
# assignPlane = array([8, 8, 8, 8])
# assignPlane = array([9, 7, 7, 9])
section = array([nodesInPlane*assignPlane[3], nodesInPlane*(assignPlane[3]+assignPlane[2]),
                nodesInPlane*(assignPlane[3]+assignPlane[2]+assignPlane[1]),
                numberOfNodes])
hMatrix = zeros(numberOfNodes)
hMatrix[0: int(section[0])] = hIni[3]
hMatrix[int(section[0]):int(section[1])] = hIni[2]
hMatrix[int(section[1]):int(section[2])] = hIni[1]
hMatrix[int(section[2]):int(section[3])] = hIni[0]
# hMatrix = -0.01*ones(numberOfNodes)


# Calculation of hydraulic conductivity
def hydraulic_conductivity(h, ks, alpha, n):
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
def capillary_capacity(h, theta_s, theta_r, alpha, n, s=S):
    cc = 0.5*(((1+np.sign(h))*s)+
              (1-np.sign(h))*(s+((theta_s-theta_r)*alpha*n*(1-1/n))*((-1*alpha*-1*((h)**2)**(0.5))**(n-1))*
                              ((1+(-1*alpha*-1*((h)**2)**(0.5))**n)**(-(2-1/n)))))
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
    for i in range(0, numberOfNodes):
        # print('time: ', timeList)
        # print('nodes: ', i)
        current_state = state[i]
        dz = dzList
        if i == 0:
            bc_zl = current_state
            # bc_zl = 0
            bc_zu = state[i + nodesInPlane]
        elif i == nodesInZ - 1:
            bc_zl = state[i - nodesInPlane]
            KzU1 = hydraulic_conductivity(current_state, ks, alpha, n)
            bc_zu = current_state + dz * (-1 + irr / KzU1)
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
    h = zeros((len(timeList), numberOfNodes))
    h[0] = hMatrix
    h0 = h[0]
    theta = zeros((len(timeList), numberOfNodes))
    h_avg = zeros((len(timeList), 4))  # 4 sensors
    h_avg[0] = hIni
    theta_avg = zeros((len(timeList), 4))
    theta_avg[0] = thetaIni*100
    for i in range(len(timeList)-1):
        print('At time:', i+1, ' min(s)')
        ts = [timeList[i], timeList[i+1]]
        y = integrate.odeint(ode, h0, ts, args=(irr_amount[i], p))
        h0 = y[-1]
        h[i+1] = h0

        theta0 = 100 * ((p[1] - p[2]) * (1 + (-p[3] * h0) ** p[4]) ** (-(1 - 1 / p[4])) + p[2])
        theta[i+1] = theta0

        start = 2
        end = 9
        for j in range(0, 4):
            h_avg[i+1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                    (end - start) * nodesInPlane))
            theta_avg[i+1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                    (end - start) * nodesInPlane))
            start += 7
            end += 7
        h_avg[i+1] = h_avg[i+1][::-1]
        theta_avg[i+1] = theta_avg[i+1][::-1]
    return h_avg, theta_avg


# Time interval
# General
ratio_t = 2
dt = 60.0  # second
timeSpan = 1444  # min # 19 hour
# timeSpan = 927
interval = int(timeSpan*60/dt)

# # Irrigation scheduled
# ratio_t = 2
# dt = 60.0/ratio_t  # second
# timeSpan = 1444  # min # 19 hour
# interval = int(timeSpan*60/dt)

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
with open('Data/exp_data_5L.dat', 'r') as f:
    whole = f.readlines()
    for index, line in enumerate(whole):
        one_line = line.rstrip().split(",")
        one_line = one_line[1:5]
        h_temp = []
        theta_temp = []
        for index1, item in enumerate(one_line):
            item = float(item)
            theta_temp.append(item)
            temp1 = ((item/100 - Theta_r) / (Theta_s - Theta_r))
            temp2 = (temp1 ** (1. / (-(1. - (1. / N)))))
            temp3 = (temp2 - 1)
            temp4 = (temp3 ** (1. / N))
            item = temp4 / (-Alpha)
            h_temp.append(item)
        h_temp = array(h_temp, dtype='O')
        theta_temp = array(theta_temp, dtype='O')
        he[index] = h_temp
        theta_e[index] = theta_temp

irr_amount = zeros((len(timeList), 1))
for i in range(0, interval):
    # irr_amount[i] = 0

    if i in range(0, 22*ratio_t):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (22*ratio_t * 60))
    elif i in range(59*ratio_t, 87*ratio_t):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27*ratio_t * 60))
    elif i in range(161*ratio_t, 189*ratio_t):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27*ratio_t * 60))
    elif i in range(248*ratio_t, 276*ratio_t):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27*ratio_t * 60))
    elif i in range(335*ratio_t, 361*ratio_t):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (25*ratio_t * 60))
    else:
        irr_amount[i] = 0

p0 = [Ks, Theta_s, Theta_r, Alpha, N]
p = [Ksp, Theta_sp,Theta_rp, Alphap, Np]
hi, theta_i = simulate(p0)
hp, theta_p = simulate(p)

hi = ((((theta_i / 100 - p0[2]) / (p0[1] - p0[2])) ** (1. / (-(1 - 1 / p0[4]))) - 1) ** (1. / p0[4])) / (
                -p0[3])
temp11 = (((theta_p / 100 - p[2]) / (p[1] - p[2]))+mini)
temp22 = temp11 ** (1. / (-(1 - 1 / p[4])))
temp33 = (temp22 - 1)
temp44 = (temp33 ** (1. / p[4]))
hp = temp44 / (-p[3])
# hp[0] = hIni[0]

plt.figure(1)
# plt.subplot(4, 1, 1)
plt.plot(timeList/60.0, hi[:, 0], 'y--', label=r'$h_1$ initial')
plt.plot(timeList_original/60.0, he[:, 0], 'b:', label=r'$h_1$ measured')
plt.plot(timeList/60.0, hp[:, 0], 'r-', label=r'$h_1$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Pressure head (m)')
plt.legend(loc='best')
plt.show()

# plt.figure(2)
# # plt.subplot(4, 1, 2)
# plt.plot(timeList/60.0, hi[:, 1], 'y:', label=r'$h_2$ initial')
# plt.plot(timeList/60.0, he[:, 1], 'b-', label=r'$h_2$ measured')
# plt.xlabel('Time (min)')
# plt.ylabel('Pressure head (m)')
# plt.legend(loc='best')
# plt.show()

# plt.figure(3)
# # plt.subplot(4, 1, 3)
# plt.plot(timeList/60.0, hi[:, 2], 'y:', label=r'$h_3$ initial')
# plt.plot(timeList/60.0, he[:, 2], 'b-', label=r'$h_3$ measured')
# plt.xlabel('Time (min)')
# plt.ylabel('Pressure head (m)')
# plt.legend(loc='best')
# plt.show()

# plt.figure(4)
# # plt.subplot(4, 1, 4)
# plt.plot(timeList/60.0, hi[:, 3], 'y:', label=r'$h_4$ initial')
# plt.plot(timeList/60.0, he[:, 3], 'b-', label=r'$h_4$ measured')
# plt.xlabel('Time (min)')
# plt.ylabel('Pressure head (m)')
# plt.legend(loc='best')
# plt.show()

plt.figure(5)
# plt.subplot(4, 1, 1)
plt.plot(timeList/60.0, theta_i[:, 0], 'y--', label=r'$theta_1$ initial')
plt.plot(timeList_original/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
plt.plot(timeList/60.0, theta_p[:, 0], 'r-', label=r'$theta_1$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

# plt.figure(6)
# # plt.subplot(4, 1, 2)
# plt.plot(timeList/60.0, theta_i[:, 1], 'y:', label=r'$theta_2$ initial')
# plt.plot(timeList/60.0, theta_e[:, 1], 'b-', label=r'$theta_2$ measured')
# plt.xlabel('Time (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()

# plt.figure(7)
# # plt.subplot(4, 1, 3)
# plt.plot(timeList/60.0, theta_i[:, 2], 'y:', label=r'$theta_3$ initial')
# plt.plot(timeList/60.0, theta_e[:, 2], 'b-', label=r'$theta_3$ measured')
# plt.xlabel('Time (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()

# plt.figure(8)
# # plt.subplot(4, 1, 4)
# plt.plot(timeList/60.0, theta_i[:, 3], 'y:', label=r'$theta_4$ initial')
# plt.plot(timeList/60.0, theta_e[:, 3], 'b-', label=r'$theta_4$ measured')
# plt.xlabel('Time (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

io.savemat('sim_results_1D_farm.mat', dict(y_1D_farm=theta_i))
