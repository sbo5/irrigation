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
Ks = 0.00000288889  # [m/s]
Theta_s = 0.43
Theta_r = 0.078
Alpha = 3.6
N = 1.56

S = 0.0001  # [per m]
PET = 0.0000000070042  # [per sec]

# Initial state
thetaIni = array([10.1, 8.4, 8.6, 10.0])/100
# thetaIni = array([26.0, 26.0, 13.0, 11.0])/100
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


def ode(x, u, p):
    # Optimized parameters
    ks, theta_s, theta_r, alpha, n = p
    irr = u
    state = x
    dhdt = SX.zeros(numberOfNodes)
    # dhdt = zeros(numberOfNodes)
    for i in range(0, numberOfNodes):
        # print('time: ', timeList)
        # print('nodes: ', i)
        current_state = state[i]
        coordinate = positionOfNodes[i]
        for index, item in enumerate(coordinate):
            if index == 0:
                pass
            elif index == 1:
                pass
            else:
                if item == 0:
                    bc_zl = current_state
                    # bc_zl = 0
                    bc_zu = state[i + nodesInPlane]
                elif item == nodesInZ - 1:
                    bc_zl = state[i - nodesInPlane]
                    KzU1 = hydraulic_conductivity(current_state, ks, alpha, n)
                    bc_zu = current_state + dz * (-1 + irr / KzU1)
                else:
                    bc_zl = state[i - nodesInPlane]
                    bc_zu = state[i + nodesInPlane]
                dz = dzList

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
    for i in range(0, len(timeList)-1):
        print('At time:', i+1, ' min(s)')
        y = model(h0, irr_amount[i], p)
        h0 = y
        h0 = h0.full()
        h0 = h0.ravel()
        h[i+1] = h0

        temp11 = (1 + (-p[3] * h0) ** p[4])
        temp22 = temp11 ** (-(1 - 1 / p[4]))
        temp33 = (p[1] - p[2]) * temp22
        theta0 = 100 * (temp33 + p[2])
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


def objective(p):
    # Simulate objective
    print(p)
    hp, theta_p = simulate(p)

    if obj_fun == 1:
        pass
    else:
        hp = ((((theta_p / 100 - p[2]) / (p[1] - p[2])) ** (1. / (-(1 - 1 / p[4]))) - 1) ** (1. / p[4])) / (
                    -p[3])

        he = ((((theta_e / 100 - p[2]) / (p[1] - p[2])) ** (1. / (-(1 - 1 / p[4]))) - 1) ** (1. / p[4])) / (
                    -p[3])

    # # Calculate objective
    obj = 0.0
    for i in range(len(timeList)):
        if obj_fun == 1:
            obj += ((theta_p[i, 0] - theta_e[i, 0])/theta_e[i, 0]) ** 2 + \
                   ((theta_p[i, 1] - theta_e[i, 1])/theta_e[i, 1]) ** 2 + \
                   ((theta_p[i, 2] - theta_e[i, 2])/theta_e[i, 2]) ** 2 + \
                   ((theta_p[i, 3] - theta_e[i, 3])/theta_e[i, 3]) ** 2
        else:
            obj += ((hp[i, 0]-he[i, 0])/he[i, 0])**2 + \
                   ((hp[i, 1]-he[i, 1])/he[i, 1])**2 + \
                   ((hp[i, 2]-he[i, 2])/he[i, 2])**2 + \
                   ((hp[i, 3]-he[i, 3])/he[i, 3])**2
    return obj


# Time interval
dt = 60.0  # second
# timeSpan = 1444  # min # 19 hour
timeSpan = 10  # 1st set is the initial condition. Its actually 9 mins
interval = int(timeSpan*60/dt)

timeList = []
for i in range(interval):
    current_t = i*dt
    timeList.append(current_t)
timeList = array(timeList, dtype='O')

he = zeros((len(timeList), 4))  # four sensors
theta_e = zeros((len(timeList), 4))
with open('exp_data.dat', 'r') as f:
    whole = f.readlines()
    for index, line in enumerate(whole):
        one_line = line.rstrip().split(",")
        one_line = one_line[1:5]
        h_temp = []
        theta_temp = []
        for index1, item in enumerate(one_line):
            item = float(item)
            theta_temp.append(item)
            temp1 = ((item / 100 - Theta_r) / (Theta_s - Theta_r))
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
    if i in range(0, 22):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (22 * 60))
        # irr_amount = 7.3999e-08
    elif i in range(59, 87):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
        # irr_amount = 7.3999e-08
    elif i in range(161, 189):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
        # irr_amount = 7.3999e-08
    elif i in range(248, 276):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
        # irr_amount = 7.3999e-08
    elif i in range(335, 361):
        irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (25 * 60))
        # irr_amount = 7.3999e-08
    else:
        irr_amount[i] = 0
        # irr_amount = 7.3999e-08

obj_fun = int(input('Enter 1 for Theta-base objective function, or 2 for h-base: '))

p0 = array([Ks, Theta_s, Theta_r, Alpha, N])
model = mpc.DiscreteSimulator(ode, dt, [numberOfNodes, 1, 5], ['x', 'u', 'p'])
print('Initial SSE Objective: ' + str(objective(p0)))
bnds = ((1e-10, 1e-3), (0.302, 1.0), (0.0, 0.084), (-inf, inf), (0.8, inf))
solution = optimize.minimize(objective, p0, bounds=bnds)
p = solution.x
print('Final SSE Objective: ' + str(objective(p)))
Ks = p[0]
Theta_s = p[1]
Theta_r = p[2]
Alpha = p[3]
N = p[4]
print('Ks: ' + str(Ks))
print('Theta_s: ' + str(Theta_s))
print('Theta_r: ' + str(Theta_r))
print('Alpha: ' + str(Alpha))
print('N: ' + str(N))

hi, theta_i = simulate(p0)
hp, theta_p = simulate(p)

hi = ((((theta_i / 100 - p0[2]) / (p0[1] - p0[2])) ** (1. / (-(1 - 1 / p0[4]))) - 1) ** (1. / p0[4])) / (
                -p0[3])
hp = ((((theta_p / 100 - p[2]) / (p[1] - p[2])) ** (1. / (-(1 - 1 / p[4]))) - 1) ** (1. / p[4])) / (
                -p[3])
hp[0] = hIni[0]

plt.figure(1)
# plt.subplot(4, 1, 1)
plt.plot(timeList/60.0, hi[:, 0], 'y--', label=r'$h_1$ initial')
plt.plot(timeList/60.0, he[:, 0], 'b:', label=r'$h_1$ measured')
plt.plot(timeList/60.0, hp[:, 0], 'r-', label=r'$h_1$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Pressure head (m)')
plt.legend(loc='best')
plt.show()

plt.figure(2)
# plt.subplot(4, 1, 2)
plt.plot(timeList/60.0, hi[:, 1], 'y--', label=r'$h_2$ initial')
plt.plot(timeList/60.0, he[:, 1], 'b:', label=r'$h_2$ measured')
plt.plot(timeList/60.0, hp[:, 1], 'r-', label=r'$h_2$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Pressure head (m)')
plt.legend(loc='best')
plt.show()

plt.figure(3)
# plt.subplot(4, 1, 3)
plt.plot(timeList/60.0, hi[:, 2], 'y--', label=r'$h_3$ initial')
plt.plot(timeList/60.0, he[:, 2], 'b:', label=r'$h_3$ measured')
plt.plot(timeList/60.0, hp[:, 2], 'r-', label=r'$h_3$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Pressure head (m)')
plt.legend(loc='best')
plt.show()

plt.figure(4)
# plt.subplot(4, 1, 4)
plt.plot(timeList/60.0, hi[:, 3], 'y--', label=r'$h_4$ initial')
plt.plot(timeList/60.0, he[:, 3], 'b:', label=r'$h_4$ measured')
plt.plot(timeList/60.0, hp[:, 3], 'r-', label=r'$h_4$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Pressure head (m)')
plt.legend(loc='best')
plt.show()

plt.figure(5)
# plt.subplot(4, 1, 1)
plt.plot(timeList/60.0, theta_i[:, 0], 'y--', label=r'$theta_1$ initial')
plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
plt.plot(timeList/60.0, theta_p[:, 0], 'r-', label=r'$theta_1$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure(6)
# plt.subplot(4, 1, 2)
plt.plot(timeList/60.0, theta_i[:, 1], 'y--', label=r'$theta_2$ initial')
plt.plot(timeList/60.0, theta_e[:, 1], 'b-', label=r'$theta_2$ measured')
plt.plot(timeList/60.0, theta_p[:, 1], 'r-', label=r'$theta_2$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure(7)
# plt.subplot(4, 1, 3)
plt.plot(timeList/60.0, theta_i[:, 2], 'y--', label=r'$theta_3$ initial')
plt.plot(timeList/60.0, theta_e[:, 2], 'b:', label=r'$theta_3$ measured')
plt.plot(timeList/60.0, theta_p[:, 2], 'r-', label=r'$theta_3$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure(8)
# plt.subplot(4, 1, 4)
plt.plot(timeList/60.0, theta_i[:, 3], 'y--', label=r'$theta_4$ initial')
plt.plot(timeList/60.0, theta_e[:, 3], 'b:', label=r'$theta_4$ measured')
plt.plot(timeList/60.0, theta_p[:, 3], 'r-', label=r'$theta_4$ optimized')
plt.xlabel('Time (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))
io.savemat('sim_results_1D_mpctools.mat', dict(y_1D_mpctools=theta_i))
