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
from numpy import dot, zeros, eye, outer
from numpy.random import randn
from numpy.random import multivariate_normal
from scipy.linalg import inv

# User selection: Nominal model, or real model?
#ans = int(input('Type 1 for nominal model, or 2 for real model: '))
#if ans == 1:
#    processNoise = 0
#    csvFilenamePlot = 'nominalmodelT_plot.csv'
#elif ans == 2:
#    processNoise = 0.05
#    csvFilenamePlot = 'realmodelT.csv'
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

#Parameters of the system
processNoise = 0
csvFileNamePlot = 'nominalmodelT_plot.csv'
csvFileNameCal = 'nominalmodelT_cal.csv'

timeSpan = 20
dt = 86400

dim_u = 1  # Number of inputs
#dim_z = 1  # Number of measurements

N = 1  # sigma points

# Initial condition
h_initial = -0.2
h_i = np.ones(nodesInZ*numberOfGrids)*h_initial
for i in range(nodesInZ):
    for j in range(nodesInX):    
        k = (j+1)*nodesInX - 1
        k = i*numberOfGrids + k
        if k < nodesInZ*numberOfGrids:
#            h_i[k-2] = -0.6
            h_i[k] = -0.8
u_i = zeros(dim_u)

# Variance / Covariance
#stdState = 0.0  # State covariance
#stdNoise = 0.0  # Process and measurement covariance
##
##
##
##

## Define EnKF
#x = np.array(h_i)
#u = np.array(u_i)
#P = np.eye(nodesInZ) * stdState * stdState

#f = EnKF(x=x, u=u, P=P, dim_z=dim_z, N=N)
#f.R *= stdNoise ** 2
#f.Q *= stdNoise ** 2
#ensemble = f.initialize(x=x, P=P)

# Generating lists for data collection and for plots
thetaList = []
ThetaListOne = []
ThetaListAll = []
ThetaListMeanT = []
ThetaListMeanT1Grid = []
ThetaListMeanT1Time = []
ThetaListMeanT1TimeForPlot = []
ThetaListMeanTAll = []
ThetaListMeanTAllForPlot = []

# Prediction using the OpenFoam -- RichardsFoam2
blockRun=BasicRunner(argv=["blockMesh"],
                     silent=True,
                     server=False)
print ("Running blockMesh")
blockRun.start()
if not blockRun.runOK():
    error("There was a problem with blockMesh")

print ("Running RichardsFoam2")
for i in range(N): # N ensemble member
    print('Ensemble number: ', i+1)
    ThetaListOne = []
    for j in range(timeSpan):
        print('At day: ', j+1)
        control["startTime"] = j*dt  
        control["endTime"] = (j+1)*dt
        control.writeFile()
        state = ParsedParameterFile(str(control["startTime"])+'/psi')
        if j == 0:
            nonuniform = []
            for k in range(len(h_i)):
                nonuniform.append(str(h_i[k])+'\n')
            state['internalField'] = '  nonuniform List<scalar>\n' +\
                                        str(len(nonuniform))+'\n' +\
                                        str('(\n') + ''.join(nonuniform)+\
                                        str(')') 
            state.writeFile()
        else:
            nonuniform1 = []
            for k in range(len(thetaList)):
                nonuniform1.append(str(thetaList[k])+'\n')
            state["internalField"] = '  nonuniform List<scalar>\n' +\
                                        str(len(nonuniform1))+'\n' +\
                                        str('(\n') + ''.join(nonuniform1)+\
                                        str(')') 
            state.writeFile()
        
                
        theRun=BasicRunner(argv=["RichardsFoam2_PF"],
                           silent=True)
        theRun.start()
        
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
# END of Calculation    

ThetaListAll = np.array(ThetaListAll)

for i in range(timeSpan):  
    meanValue = zeros(len(thetaList))
    k = i
    for j in range(N):
        meanValue += ThetaListAll[k]
        k += timeSpan
    meanValue = meanValue/N    
    ThetaListMeanT.append(meanValue)

for i in range(timeSpan):
    ThetaListMeanT1TimeForPlot = []
    for j in range(numberOfGrids):
        l = j
        ThetaListMeanT1Grid = []
        for k in range(nodesInZ):    
            value = ThetaListMeanT[i][l]
            ThetaListMeanT1Grid.append(value)
            l += numberOfGrids
        ThetaListMeanT1GridR = ThetaListMeanT1Grid[::-1]
        ThetaListMeanT1TimeForPlot.append(ThetaListMeanT1GridR)
    ThetaListMeanTAllForPlot.extend(ThetaListMeanT1TimeForPlot)

#
depthList = []
for i in range(0,depth*nodesInZ):
    depthList.append(-1*i*dz)

# Write to csv file
with open(csvFileNamePlot, 'w') as csvFile:
    table = csv.writer(csvFile)
#    table.writerow(depthList)
    for i in range(timeSpan*numberOfGrids):
        table.writerow(ThetaListMeanTAllForPlot[i])
        
# Write to csv file
with open(csvFileNameCal, 'w') as csvFile:
    table = csv.writer(csvFile)
#    table.writerow(depthList)
    for i in range(timeSpan):
        table.writerow(ThetaListMeanT[i])

##Plot the graph
#plt.plot(x, depthList, label='Initial profile')
#plt.xlabel('Soil pressuren head, h (m)')
#plt.ylabel('Soil depth, z (m)')
#plt.title('Initial soil pressure head profile')
#plt.legend()
#plt.show()

for n in range(5):  
    print('Grid number: ', n+1)
    for i in range(timeSpan):
        plt.plot(ThetaListMeanTAllForPlot[i*numberOfGrids+n], depthList, label='Prediction '+str(i+1)+' day')
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