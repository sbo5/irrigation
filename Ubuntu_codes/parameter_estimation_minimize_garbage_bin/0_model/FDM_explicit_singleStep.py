from __future__ import (absolute_import, print_function, division, unicode_literals)
import scipy.io
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

# Parameters
Ks = 0.00000288889  # [m/s]
Theta_s = 0.43
Theta_r = 0.078
Alpha = 3.6
N = 1.56
S = 0.0001  # [per m]
PET = 0.0000000070042  # [per sec]

# Label the nodes
indexOfNodes = []
for i in range(0, numberOfNodes):
    indexOfNodes.append(i)
positionOfNodes = []
for k in range(0, nodesInZ):
    for j in range(0, nodesInY*4):
        for i in range(0, nodesInX):
            positionOfNodes.append([i, j, k])

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
x1 = zeros(numberOfNodes)


# Calculation of hydraulic conductivity
def hydraulic_conductivity(h, ks=Ks, alpha=Alpha, n=N):
    term3 = (1+((-1*alpha*-1*((h)**2)**(1./2.))**n))
    term4 = (term3**(-(1-1/n)))
    term5 = term4**(1./2.)
    term6 = term4**(n/(n-1))
    term7 = (1-term6)**(1-1/n)
    term8 = ((1-term7)**2)
    term1 = ((1 + sign(h)) * ks)
    term2 = (1-sign(h))*ks*(term5)*term8
    term0 = (term1+term2)
    hc = 0.5*term0
    return hc


# Calculation of capillary capacity
def capillary_capacity(h, s=S, theta_s=Theta_s, theta_r=Theta_r, alpha=Alpha, n=N):
    cc = 0.5*(((1+np.sign(h))*s)+
              (1-np.sign(h))*(s+((theta_s-theta_r)*alpha*n*(1-1/n))*((-1*alpha*-1*((h)**2)**(0.5))**(n-1))*
                              ((1+(-1*alpha*-1*((h)**2)**(0.5))**n)**(-(2-1/n)))))
    return cc


def mean_hydra_conductivity(left_boundary, right_boundary):
    lk = hydraulic_conductivity(left_boundary)
    rk = hydraulic_conductivity(right_boundary)
    mk = mean([lk, rk])
    return mk


# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def ode(x, u):
    irr = u
    head_pressure = SX.zeros(numberOfNodes)
    for i in range(0, numberOfNodes):
        state = x[i]
        coordinate = positionOfNodes[i]
        for index, item in enumerate(coordinate):
            if index == 0:
                if item == 0:
                    bc_xl = state
                    # bc_xl = 0
                    bc_xr = x[i+1]
                    dxl = dxList[0]
                    dxr = dxList[1]
                elif item == nodesInX-1:
                    bc_xl = x[i - 1]
                    bc_xr = state
                    # bc_xr = 0
                    dxl = dxList[2]
                    dxr = dxList[2]
                else:
                    bc_xl = x[i - 1]
                    bc_xr = x[i + 1]
                    if item == 1:
                        dxl = dxList[1]
                        dxr = dxList[2]
                    else:
                        dxl = dxList[2]
                        dxr = dxList[2]
            elif index == 1:
                if item == 0:
                    bc_yl = x[i + nodesInX*(nodesInY*4-1)]
                    # bc_yl = 0
                    bc_yr = x[i + nodesInX]
                elif item == nodesInY*4-1:
                    bc_yl = x[i - nodesInX]
                    bc_yr = x[i - nodesInX*(nodesInY*4-1) ]
                    # bc_yr = 0
                else:
                    bc_yl = x[i - nodesInX]
                    bc_yr = x[i + nodesInX]
                remainder = i % nodesInX
                dy = dyList[remainder]
            else:
                if item == 0:
                    bc_zl = state
                    # bc_zl = 0
                    bc_zu = x[i + nodesInPlane]
                elif item == nodesInZ-1:
                    bc_zl = x[i - nodesInPlane]
                    KzU1 = hydraulic_conductivity(state)
                    bc_zu = state + dz*(-1 + irr/KzU1)
                else:
                    bc_zl = x[i - nodesInPlane]
                    bc_zu = x[i + nodesInPlane]
                dz = dzList

        KxL = mean_hydra_conductivity(x[i], bc_xl)
        KxR = mean_hydra_conductivity(x[i], bc_xr)
        deltaHxL = (x[i] - bc_xl) / dxl
        deltaHxR = (bc_xr - x[i]) / dxr

        KyL = mean_hydra_conductivity(x[i], bc_yl)
        KyR = mean_hydra_conductivity(x[i], bc_yr)
        deltaHyL = (x[i] - bc_yl) / dy
        deltaHyR = (bc_yr - x[i]) / dy

        KzL = mean_hydra_conductivity(x[i], bc_zl)
        KzU = mean_hydra_conductivity(x[i], bc_zu)
        deltaHzL = (x[i] - bc_zl) / dz
        deltaHzU = (bc_zu - x[i]) / dz

        temp0 = 1 / (0.5 * (dxr+dxl)) * (KxR * deltaHxR - KxL * deltaHxL)
        temp1 = 1 / (0.5 * 2 * dy) * (KyR * deltaHyR - KyL * deltaHyL)
        temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
        temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
        temp4 = 0  # source term
        temp5 = temp0 + temp1 + temp2 + temp3 - temp4
        temp6 = temp5 / capillary_capacity(state)
        head_pressure[i] = temp6
    return head_pressure


# Time interval
# KIni = hydraulic_conductivity(hIni)
# CIni = capillary_capacity(hIni)
# dt = 0.5*dz*dz/(KIni/CIni)*3
dt = 60  # second
timeSpan = 1444  # min # 19 hour
# timeSpan = 600
interval = int(timeSpan*60/dt)

solList_theta = []
solList_h = []
solList_thetaAvg = []

model = mpc.DiscreteSimulator(ode, dt, [numberOfNodes, 1], ['x', 'u'])

for t in range(0, interval):
    if t in range(0, 22):
        irr_amount = (0.001/(3.1415*0.22*0.22)/(22*60))
        # irr_amount = 7.3999e-08
    elif t in range(59, 87):
        irr_amount = (0.000645/(pi*0.22*0.22)/(27*60))
        # irr_amount = 7.3999e-08
    elif t in range(161, 189):
        irr_amount = (0.000645/(pi*0.22*0.22)/(27*60))
        # irr_amount = 7.3999e-08
    elif t in range(248, 276):
        irr_amount = (0.000645/(pi*0.22*0.22)/(27*60))
        # irr_amount = 7.3999e-08
    elif t in range(335, 361):
        irr_amount = (0.000645 / (pi * 0.22 * 0.22) / (25 * 60))
        # irr_amount = 7.3999e-08
    else:
        irr_amount = 0
        # irr_amount = 7.3999e-08
    # irr_amount = 7.3999e-08
    print('At time: ', t + 1, 'min(s)')

    x1 = model(hMatrix, irr_amount)

    thetaList = ones(numberOfNodes)  # re-initialize thetaList
    for i in range(0, numberOfNodes):
        if x1[i] >= 0:
            theta = Theta_s
        else:
            theta = (Theta_s-Theta_r)*(1 + (-Alpha*x1[i])**N)**(-(1-1/N)) + Theta_r

        thetaList[i] = theta
    thetaList = np.asarray(thetaList)
    solList_theta.append(thetaList)

    thetaListAvg = ones(4)  # 4 sensors
    start = 2
    end = 9
    for i in range(0, 4):
        thetaListAvg[i] = (sum(thetaList[start*nodesInPlane:end*nodesInPlane])/((end-start)*nodesInPlane))*100
        start += 7
        end += 7
    thetaListAvgRev = thetaListAvg[::-1]
    solList_thetaAvg.append(thetaListAvgRev)

    hMatrix = copy(x1)
    solList_h.append(x1)
solList_thetaAvg = np.asarray(solList_thetaAvg)

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

scipy.io.savemat('sim_results_form1.mat', dict(y=solList_thetaAvg))
