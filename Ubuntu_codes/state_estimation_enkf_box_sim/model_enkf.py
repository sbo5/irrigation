#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:03:54 2018

@author: sbo
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import time
from PyFoam.Error import error
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.common import Q_discrete_white_noise
from numpy import dot, zeros, ones, eye, outer
from numpy.random import randn
from numpy.random import multivariate_normal
from scipy.linalg import inv

# User selection: nominal model + EnKF or real model + EnKF
#ans = int(input('Type 1 for (nominalmodel + EnKF), or 2 for (realmodel + EnKF): '))
#if ans == 1:
#    csvFileNameT = 'nominalmodelT.csv'
#    csvFileNameF = 'nominalmodelF.csv'
#    std_noise_Q = 0.
#    std_noise_R = 0.0025

#elif ans == 2:
#    csvFileNameT = 'realmodel.csv'
#    # User selection: Large Q/R (process noise/measurement noise) or Large R/Q?
#    ans = int(input('Type 1 for large process noise (small measurement noise), or 2 '
#                    'for small process noise (large measurement noise): '))
#    if ans == 1:
#        Q_to_R = 100
#        std_noise_R = 0.001  # measurement noise
#        std_noise_Q = std_noise_R * Q_to_R  # process noise
#    elif ans == 2:
#        R_to_Q = 100
#        std_noise_Q = 0.001  # process noise
#        std_noise_R = std_noise_Q * R_to_Q  # measurements noise
#    else:
#        print('Invalid input')
#else:
#    print('Invalid input')

##
##
##
start_time = time.clock()

# Change the time precision written in OpenFoam
control = ParsedParameterFile('system/controlDict')
control['timePrecision'] = 10
control.writeFile()
# Change the blockMeshDict
lengthOfField = 5
dx = 1
nodesInX = int(lengthOfField/dx)

depth = 1
dz = 0.02
nodesInZ = int(depth/dz)

numberOfGrids = nodesInX * nodesInX
#numberOfGrids = 1
mesh = ParsedParameterFile('constant/polyMesh/blockMeshDict')
mesh['vertices'] = [[0, 0, 0],
                    [lengthOfField, 0, 0],
                    [lengthOfField, lengthOfField, 0],
                    [0, lengthOfField, 0],
                    [0, 0, depth],
                    [lengthOfField, 0, depth],
                    [lengthOfField, lengthOfField, depth],
                    [0, lengthOfField, depth]]
mesh['blocks'] = ['hex (0 1 2 3 4 5 6 7)         ('+str(nodesInX)+' '+str(nodesInX)+' '+str(nodesInZ)+') simpleGrading (1 1 1)']
mesh.writeFile()

# Read data from CSV files
csvFileNameTPlot = 'nominalmodelT_plot.csv'
csvFileNameTCal = 'nominalmodelT_cal.csv'
csvFileNameF = 'nominalmodelF.csv'
std_noise_Q = 0.0
std_noise_R = 0.0137

# Parameters of the system
timeSpan = 20 # Time
dt = 86400

dim_u = 1  # Number of inputs
#dim_z = 1  # Number of measurements

N = 10  # sigma points

# Initial condition
#h_initialTrue = -0.01
#h_initialTrue1 = -0.99
h_initial = -0.5
#h_iTrue = np.ones(nodesInZ)*h_initialTrue
#h_iTrue1 = np.ones(nodesInZ)*h_initialTrue1
h_i = np.ones(nodesInZ*numberOfGrids)*h_initial
u_i = zeros(dim_u)

# Variance / Covariance
#stdState = 0.15811  # State covariance
stdState = 0.025  # State covariance
stdNoise = 0.0  # Process and measurement covariance

# h function, which maps states matrix into predicted measurements
depthOfSensor = [5, 10,15, 20,30,35,40, 45]
H = zeros(nodesInZ*numberOfGrids)
if depthOfSensor == []:
    H = ones(nodesInZ*numberOfGrids)

for item in depthOfSensor:
    node = nodesInZ - 1 - item
    slicePosition = node*numberOfGrids
    H[node*numberOfGrids: (node+1)*numberOfGrids] = 1
#H = np.ones(nodesInZ*numberOfGrids)

# Number of measurements
dim_z = 0
for element in H:
    if element != 0:
        dim_z += 1
##
##
##
        
# Define EnKF
#xTrue = np.array(h_iTrue)
#xTrue1 = np.array(h_iTrue1)
x = np.array(h_i)
u = np.array(u_i)
P = np.eye(nodesInZ*numberOfGrids) * stdState * stdState

f = EnKF(x=x, u=u, P=P, dim_z=dim_z, N=N)
f.R *= std_noise_R ** 2
f.Q *= std_noise_Q ** 2
ensemble = f.initialize(x=x, P=P)

# Generating lists for data collection and for plots
thetaList = []
ThetaListOne = []
ThetaListAll = []

ThetaU = []
ThetaUR = []
ThetaListMeanU = []
measurementListTPlot = []
measurementListTCal = []
measurementListF = []
measurementSensorList = []
ThetaListMeanUForPlot = []
ThetaListMeanUAllForPlot = []

resultsTCal = []
with open(csvFileNameTCal) as csvFile:
    readCsv = csv.reader(csvFile)
    for row in readCsv:
        resultsTCal.append(row)
for i in range(len(resultsTCal)):
    measurementListTCal.append(resultsTCal[i])
    for index, item in enumerate(measurementListTCal[i]):
        measurementListTCal[i][index] = float(item)    
        
resultsTPlot = []
with open(csvFileNameTPlot) as csvFile:
    readCsv = csv.reader(csvFile)
    for row in readCsv:
        resultsTPlot.append(row)
for i in range(len(resultsTPlot)):
    measurementListTPlot.append(resultsTPlot[i])
    for index, item in enumerate(measurementListTPlot[i]): # Actually elements in resultTPlot are floats 
        measurementListTPlot[i][index] = float(item)
        
resultsF = []
with open(csvFileNameF) as csvFile:
    readCsv = csv.reader(csvFile)
    for row in readCsv:
        resultsF.append(row)
for i in range(len(resultsF)):
    measurementListF.append(resultsF[i])
    for index, item in enumerate(measurementListF[i]):
        measurementListF[i][index] = float(item)

## Change the time precision written in OpenFoam
#control = ParsedParameterFile('system/controlDict')
#control['timePrecision'] = 10

# Prediction using the model
blockRun=BasicRunner(argv=["blockMesh"],
                     silent=True,
                     server=False)
print ("Running blockMesh")
blockRun.start()
if not blockRun.runOK():
    error("There was a problem with blockMesh")
    
print ("Running RichardsFoam2")
#for n in range(numberOfGrids):
#    print('Grids number: ', n+1)
for k in range(timeSpan): # N ensemble member  
    print('EnKF 7')
    print('At day: ', k+1)
    ThetaListAll = []
    for i in range(N):
        print('Ensemble number: ', i+1)
        ThetaListOne = [] 
        for j in range(1):
            control["startTime"] = k*86400  
            control["endTime"] = (k+1)*86400
            
            control.writeFile()

            state = ParsedParameterFile(str(control["startTime"])+'/psi')
            if k == 0:    
                nonuniform = []
                for l in range(len(ensemble[i])):
                    nonuniform.append(str(ensemble[i, l])+'\n')
                state["internalField"] = '  nonuniform List<scalar>\n' +\
                                        str(len(nonuniform))+'\n' +\
                                        str('(\n') + ''.join(nonuniform)+\
                                        str(')') 
                state.writeFile()
            else:
                nonuniform1 = []
                Positive = []
                for m in range(len(ThetaU[i])):
                    nonuniform1.append(str(ThetaU[i][m])+'\n')
                state["internalField"] = '  nonuniform List<scalar>\n' +\
                                        str(len(nonuniform1))+'\n' +\
                                        str('(\n') + ''.join(nonuniform1)+\
                                        str(')') 
                for index, item in enumerate(nonuniform1):
                    if float(item) >= 0:
                        Positive.append([index, item])
                state.writeFile()

            theRun=BasicRunner(argv=["RichardsFoam2_PF"],
                               silent=True)
            theRun.start()
            print('At day: ', k+1)
            
            thetaList = []
            c = False
            c0 = 1
            Theta = open(str(control["endTime"])+'/psi', 'r')
            csvReader = csv.reader(Theta)
            for line in Theta:
                if line == '(\n':
                    c = True
                    c0 = 0
                if line in [')\n', ');', ');\n']:
                    c = False
                    break
                if c == True & c0 == 1:
                    line.rstrip()
                    line_float = float(line)
                    thetaList.append(line_float)    
                c0 = 1      
            ThetaListOne.append(thetaList)
        ThetaListAll.extend(ThetaListOne)
# End of calculation
        
    ThetaListAll = np.array(ThetaListAll)
#    meanValueP = np.mean(ThetaListAll, axis=0)
#    ThetaListMean.append(meanValueP)
    

    measurement = measurementListTCal[k]
    measurementArray = np.array(measurement)
    measurementArray = measurementArray + randn()*std_noise_R
    if dim_z == 1: # seems useless. dim_z ==1 only happens when working with 1D EnKF
        measurementSensor = measurementArray[depthOfSensor]
    elif dim_z == nodesInZ*numberOfGrids:
        measurementSensor = measurementArray
    else:
        l = 0
        measurementSensor = np.ones(dim_z)
        for index, item in enumerate(H):
            if item != 0:
                measurementSensor[l] = measurementArray[index]
                l += 1
    measurementSensorList.append(measurementSensor)
    ThetaU = f.update(ThetaListAll, measurementSensor, H)
#    ThetaUR = []
#    for i in range(len(ThetaU)):
#        ThetaU1 = ThetaU[i]#[::-1]
#        ThetaUR.append(ThetaU1)
    ThetaMeanU = np.mean(ThetaU, axis=0)
    ThetaListMeanU.append(ThetaMeanU)


# depthList
depthList = []
for i in range(0,depth*nodesInZ):
    depthList.append(-1*i*dz)
xTrue1 = ones(nodesInZ)*-0.2
xTrue5 = ones(nodesInZ)*-0.8
xF = ones(nodesInZ)*-0.5

for i in range(timeSpan):
    ThetaListMeanUForPlot = []
    for j in range(numberOfGrids):
        l = j
        ThetaListMeanT1Grid = []
        for k in range(nodesInZ):    
            value = ThetaListMeanU[i][l]
            ThetaListMeanT1Grid.append(value)
            l += numberOfGrids
        ThetaListMeanT1GridR = ThetaListMeanT1Grid[::-1]
        ThetaListMeanUForPlot.append(ThetaListMeanT1GridR)
    ThetaListMeanUAllForPlot.extend(ThetaListMeanUForPlot)

## ThetaListMeanU
#plotU1Day = []
#plotUAll = []
#for i in range(timeSpan):
#    plotU1Day = []
#    for j in range(numberOfGrids):
#        plotU = ThetaListMeanU[i][j*nodesInZ:((j+1)*nodesInZ)]
#        plotU1Day.append(plotU)
#    plotUAll.extend(plotU1Day)

## measurementListF
#plotF1Day = []
#plotFAll = []
#for i in range(timeSpan):
#    plotF1Day = []
#    for j in range(numberOfGrids):
#        plotF = measurementListF[i][j*nodesInZ:((j+1)*nodesInZ)]
#        plotF1Day.append(plotF)
#    plotFAll.extend(plotF1Day)
    
## measurementListTCal
#plotT1Day = []
#plotTAll = []
#for i in range(timeSpan):
#    plotT1Day = []
#    for j in range(numberOfGrids):
#        plotT = measurementListTCal[i][j*nodesInZ:((j+1)*nodesInZ)]
#        plotT1Day.append(plotT)
#    plotTAll.extend(plotT1Day)


##Plot the graph
#plt.plot(xTrue, depthList, linestyle = '--', label='Process')
#plt.plot(x, depthList, label='Initial guess')
#plt.xlabel('Soil pressure head, h (m)')
#plt.ylabel('Soil depth, z (m)')
#plt.title('Initial pressure head profile')
#plt.legend()
#plt.show()


#for n in range(numberOfGrids):  
#    print('Grid number: ', n)
#    for i in range(timeSpan):
#        #f, axarr = plt.subplots(timeSpan, 2, sharey = 'depthList')
#        plt.plot(measurementListTPlot[i*numberOfGrids+n], depthList, linestyle = '--', label='Process '+str(i+1)+' day')
#        plt.plot(measurementListF[i*numberOfGrids+n],  depthList, label='Openloop prediction '+str(i+1)+' day')
#        plt.plot(ThetaListMeanUAllForPlot[i*numberOfGrids+n],  depthList, label='EnKF update '+str(i+1)+' day')
#        #plt.scatter(measurementSensorList[i], depthList[depthOfSensor], label='Sensor '+str(i+1)+' day')
#        plt.xlabel('Soil pressure head, h (m)')
#        plt.ylabel('Soil depth, z (m)')
#        plt.title('Predicted soil pressure head profile')
#        plt.legend()
#        plt.show()
#
#    try:
#        input("Press enter to continue")
#    except SyntaxError:
#        pass
    
for n in range(5):
    print('Grid number: ', n+1)
    if n == 0:
        plt.plot(xTrue1, depthList, linestyle = '--', label='Process')
        plt.plot(xF, depthList, label='Initial guess')
        plt.xlabel('Soil pressure head, h (m)')
        plt.ylabel('Soil depth, z (m)')
        plt.title('Initial pressure head profile')
        plt.legend()
        plt.show()
    if n == 4:
        plt.plot(xTrue5, depthList, linestyle = '--', label='Process')
        plt.plot(xF, depthList, label='Initial guess')
        plt.xlabel('Soil pressure head, h (m)')
        plt.ylabel('Soil depth, z (m)')
        plt.title('Initial pressure head profile')
        plt.legend()
        plt.show()  
    for i in range(timeSpan):
        #f, axarr = plt.subplots(timeSpan, 2, sharey = 'depthList')
        plt.plot(measurementListTPlot[i*numberOfGrids+n], depthList, linestyle = '--', label='Process '+str(i+1)+' day')
        plt.plot(measurementListF[i*numberOfGrids+n],  depthList, label='Openloop prediction '+str(i+1)+' day')
        plt.plot(ThetaListMeanUAllForPlot[i*numberOfGrids+n],  depthList, label='EnKF update '+str(i+1)+' day')
        #plt.scatter(measurementSensorList[i], depthList[depthOfSensor], label='Sensor '+str(i+1)+' day')
        plt.xlabel('Soil pressure head, h (m)')
        plt.ylabel('Soil depth, z (m)')
        plt.title('Predicted soil pressure head profile')
        plt.legend()
        plt.show()

#    try:
#        input("Press enter to continue")
#    except SyntaxError:
#        pass

print((time.clock() - start_time)/60, 'minutes')                   
   
    

    


