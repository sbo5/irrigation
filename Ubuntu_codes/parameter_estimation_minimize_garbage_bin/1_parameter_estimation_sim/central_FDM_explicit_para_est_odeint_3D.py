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
lengthOfZ = 0.67
# Define the nodes
nodesInX = 3  # minimum is 3!!! Since dx is designed based on this
nodesInY = 3
nodesInZ = 32
nodesInZinOF = nodesInZ+1


nodesInS1PerLayer = nodesInX*(nodesInY+1)
nodesInS2PerLayer = nodesInX*nodesInY
nodesInS3PerLayer = nodesInX*nodesInY
nodesInS4PerLayer = nodesInX*(nodesInY-1)

# These are the same as the number in OF/points
nodesInS1InOF = nodesInS1PerLayer*nodesInZinOF + nodesInZinOF  # additional nodesInZinOF are center nodes
nodesInS2InOF = nodesInS2PerLayer*nodesInZinOF
nodesInS3InOF = nodesInS3PerLayer*nodesInZinOF
nodesInS4InOF = nodesInS4PerLayer*nodesInZinOF

# There are different with the number in OF/points
nodesInPlane = nodesInX*nodesInY*4  # 4 means 4 sections
numberOfNodes = nodesInPlane*nodesInZ


# Function for finding the coordinates of centroid of the triangle
def centroid(point1, point2):
    midpoint = (point1 + point2)/2  # step 1: find the midpoint of one of the sides
    cent = (2/3)*midpoint
    cent[2] = point1[2]
    return cent


# Function for finding the coordinates of center of the quadrilateral
def center_quadrilateral(point1, point2, point3, point4):
    mid = (point1 + point2 + point3 + point4)/4
    return mid


# # Change blockMesh
# mesh = ParsedParameterFile('constant/polyMesh/blockMeshDict')
# mesh['blocks'] = ['hex (0 1 2 0 3 4 5 3) ('+str(nodesInX)+' '+str(nodesInY)+' '+str(nodesInZ)+')  simpleGrading (1 1 1)\n'
#                   'hex (0 2 7 0 3 5 6 3) ('+str(nodesInX)+' '+str(nodesInY)+' '+str(nodesInZ)+')  simpleGrading (1 1 1)\n'
#                   'hex (0 7 9 0 3 6 8 3) ('+str(nodesInX)+' '+str(nodesInY)+' '+str(nodesInZ)+')  simpleGrading (1 1 1)\n'
#                   'hex (0 9 1 0 3 8 4 3) ('+str(nodesInX)+' '+str(nodesInY)+' '+str(nodesInZ)+')  simpleGrading (1 1 1)']
# mesh.writeFile()
# # Running blockMesh
# blockRun = BasicRunner(argv=["blockMesh"],
#                        silent=True,
#                        server=False)
# print ("Running blockMesh")
# blockRun.start()
# # if not blockRun.runOK():
# #     error("There was a problem with blockMesh")

# Read mesh information -- points from OF
pointsList = []
c = False
c0 = 1
Points = open('constant/polyMesh/points', 'r')
# csvReader = csv.reader(Points)
for line in Points:
    if line == '(\n':
        c = True
        c0 = 0
    if line in [')\n', ');', ');\n']:
        c = False
        break
    if c == True & c0 == 1:
        line = line.strip('(')
        line = line.strip(')\n')
        line_split = line.split()
        line_list = [float(line_split[0]), float(line_split[1]), float(line_split[2])]
        line_array = asarray(line_list)
        pointsList.append(line_array)
    c0 = 1
pointsArray = asarray(pointsList)

# Rearrange the point array
pointsInOrder = zeros((nodesInZ+1, nodesInPlane, 3))  # 3 means x, y, and z
for i in range(0, 4):  # 4 sections
    if i == 0:  # 1st section
        for j in range(0, nodesInZ+1):
            start0 = 0  # This is for pointsInOrder
            end0 = nodesInS1PerLayer
            start00 = 1 + j*(nodesInS1PerLayer+1)  # This is for pointsArray
            end00 = (nodesInS1PerLayer + 1) + j*(nodesInS1PerLayer+1)
            pointsInOrder[j][start0:end0] = pointsArray[start00:end00]
    elif i == 1:
        for j in range(0, nodesInZ+1):
            start1 = nodesInS1PerLayer
            end1 = nodesInS1PerLayer + nodesInS2PerLayer
            start11 = nodesInS1InOF + j*nodesInS2PerLayer
            end11 = (nodesInS1InOF + nodesInS2PerLayer) + j*nodesInS2PerLayer
            pointsInOrder[j][start1:end1] = pointsArray[start11:end11]
    elif i == 2:
        for j in range(0, nodesInZ+1):
            start2 = nodesInS1PerLayer + nodesInS2PerLayer
            end2 = nodesInS1PerLayer + nodesInS2PerLayer + nodesInS3PerLayer
            start22 = nodesInS1InOF + nodesInS2InOF + j*nodesInS3PerLayer
            end22 = (nodesInS1InOF + nodesInS2InOF + nodesInS3PerLayer) + j*nodesInS3PerLayer
            pointsInOrder[j][start2:end2] = pointsArray[start22:end22]
    else:
        for j in range(0, nodesInZ+1):
            start3 = nodesInS1PerLayer + nodesInS2PerLayer + nodesInS3PerLayer
            end3 = nodesInS1PerLayer + nodesInS2PerLayer + nodesInS3PerLayer + nodesInS4PerLayer
            start33 = nodesInS1InOF + nodesInS2InOF + nodesInS3InOF + j*nodesInS4PerLayer
            end33 = (nodesInS1InOF + nodesInS2InOF + nodesInS3InOF + nodesInS4PerLayer) + j*nodesInS4PerLayer
            pointsInOrder[j][start3:end3] = pointsArray[start33:end33]

pointsOrderInOF = zeros((nodesInZ, nodesInPlane, 3))
for i in range(0, nodesInZ):  # per layer
    for j in range(0, nodesInPlane):
        if j % nodesInX == 0:
            if j // nodesInX != (nodesInY*4-1):
                pointsOrderInOF[i][j] = centroid(pointsInOrder[i][j], pointsInOrder[i][j+nodesInX])
                pointsOrderInOF[i][j][2] = pointsInOrder[i][j][2]+lengthOfZ/(nodesInZ*2)
            else:
                pointsOrderInOF[i][j] = centroid(pointsInOrder[i][j], pointsInOrder[i][0])
                pointsOrderInOF[i][j][2] = pointsInOrder[i][j][2]+lengthOfZ/(nodesInZ*2)
                thepoint = j
        else:
            if j // nodesInX != (nodesInY*4-1):
                pointsOrderInOF[i][j] = center_quadrilateral(pointsInOrder[i][j], pointsInOrder[i][j-1],
                                                        pointsInOrder[i][j+nodesInX-1], pointsInOrder[i][j+nodesInX])
                pointsOrderInOF[i][j][2] = pointsInOrder[i][j][2]+lengthOfZ/(nodesInZ*2)
            else:
                pointsOrderInOF[i][j] = center_quadrilateral(pointsInOrder[i][j], pointsInOrder[i][j-1],
                                                         pointsInOrder[i][j-thepoint-1], pointsInOrder[i][j-thepoint])
                pointsOrderInOF[i][j][2] = pointsInOrder[i][j][2]+lengthOfZ/(nodesInZ*2)

dxList = zeros(nodesInX)
dxList[0] = sqrt(pointsOrderInOF[0][0][0]**2 + pointsOrderInOF[0][0][1]**2)*2
for i in range(1, nodesInX):
    dxList[i] = sqrt((pointsOrderInOF[0][i][0]-pointsOrderInOF[0][i-1][0])**2+(pointsOrderInOF[0][i][1]-pointsOrderInOF[0][i-1][1])**2)
dyList = zeros(nodesInX)
for i in range(0, nodesInX):
    dyList[i] = sqrt((pointsOrderInOF[0][i][0]-pointsOrderInOF[0][i+nodesInX][0])**2+(pointsOrderInOF[0][i][1]-pointsOrderInOF[0][i+nodesInX][1])**2)
dzList = lengthOfZ/nodesInZ

# Label the nodes
indexOfNodes = []
for i in range(0, numberOfNodes):
    indexOfNodes.append(i)
positionOfNodes = []
for k in range(0, nodesInZ):
    for j in range(0, nodesInY*4):
        for i in range(0, nodesInX):
            positionOfNodes.append([i, j, k])

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
    #     dhdt = SX.zeros(numberOfNodes)
    dhdt = zeros(numberOfNodes)
    for i in range(0, numberOfNodes):
        # print('time: ', timeList)
        # print('nodes: ', i)
        current_state = state[i]
        coordinate = positionOfNodes[i]
        for index, item in enumerate(coordinate):
            if index == 0:
                if item == 0:
                    bc_xl = current_state
                    # bc_xl = 0
                    bc_xr = state[i+1]
                    dxl = dxList[0]
                    dxr = dxList[1]
                elif item == nodesInX-1:
                    bc_xl = state[i - 1]
                    bc_xr = current_state
                    # bc_xr = 0
                    dxl = dxList[2]
                    dxr = dxList[2]
                else:
                    bc_xl = state[i - 1]
                    bc_xr = state[i + 1]
                    if item == 1:
                        dxl = dxList[1]
                        dxr = dxList[2]
                    else:
                        dxl = dxList[2]
                        dxr = dxList[2]
            elif index == 1:
                if item == 0:
                    bc_yl = state[i + nodesInX*(nodesInY*4-1)]
                    # bc_yl = 0
                    bc_yr = state[i + nodesInX]
                elif item == nodesInY*4-1:
                    bc_yl = state[i - nodesInX]
                    bc_yr = state[i - nodesInX*(nodesInY*4-1) ]
                    # bc_yr = 0
                else:
                    bc_yl = state[i - nodesInX]
                    bc_yr = state[i + nodesInX]
                remainder = i % nodesInX
                dy = dyList[remainder]
            else:
                if item == 0:
                    bc_zl = current_state
                    # bc_zl = 0
                    bc_zu = state[i + nodesInPlane]
                elif item == nodesInZ-1:
                    bc_zl = state[i - nodesInPlane]
                    KzU1 = hydraulic_conductivity(current_state, ks, alpha, n)
                    bc_zu = current_state + dz*(-1 + irr/KzU1)
                else:
                    bc_zl = state[i - nodesInPlane]
                    bc_zu = state[i + nodesInPlane]
                dz = dzList

        KxL = mean_hydra_conductivity(state[i], bc_xl, ks, alpha, n)
        KxR = mean_hydra_conductivity(state[i], bc_xr, ks, alpha, n)
        deltaHxL = (state[i] - bc_xl) / dxl
        deltaHxR = (bc_xr - state[i]) / dxr

        KyL = mean_hydra_conductivity(state[i], bc_yl, ks, alpha, n)
        KyR = mean_hydra_conductivity(state[i], bc_yr, ks, alpha, n)
        deltaHyL = (state[i] - bc_yl) / dy
        deltaHyR = (bc_yr - state[i]) / dy

        KzL = mean_hydra_conductivity(state[i], bc_zl, ks, alpha, n)
        KzU = mean_hydra_conductivity(state[i], bc_zu, ks, alpha, n)
        deltaHzL = (state[i] - bc_zl) / dz
        deltaHzU = (bc_zu - state[i]) / dz

        temp0 = 1 / (0.5 * (dxr+dxl)) * (KxR * deltaHxR - KxL * deltaHxL)
        temp1 = 1 / (0.5 * 2 * dy) * (KyR * deltaHyR - KyL * deltaHyL)
        temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
        temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
        temp4 = 0  # source term
        temp5 = temp0 + temp1 + temp2 + temp3 - temp4
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
timeSpan = 1444  # min # 19 hour
# timeSpan = 10
interval = int(timeSpan*60/dt)

timeList = []
for i in range(interval):
    current_t = i*dt
    timeList.append(current_t)
timeList = array(timeList, dtype='O')

he = zeros((len(timeList), 4))  # four sensors
theta_e = zeros((len(timeList), 4))
with open('exp_data_original.dat', 'r') as f:
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
print('Initial SSE Objective: ' + str(objective(p0)))
bnds = ((1e-10, 1e-2), (0.302, 1.0), (0.0, 0.084), (-inf, inf), (0.8, inf))
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
#
# io.savemat('sim_results_3D_odeint.mat', dict(y_3D_odeint=theta_i))
