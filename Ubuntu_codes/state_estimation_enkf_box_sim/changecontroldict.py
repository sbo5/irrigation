#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:03:12 2018

@author: sbo
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from PyFoam.Error import error
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from numpy.random import randn
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import inv
from numpy import dot, zeros, eye, outer
from numpy.random import multivariate_normal
from filterpy.common import dot3

# User selection: nominal model + EnKF or real model + EnKF
ans = int(input('Type 1 for (nominal model + EnKF), or 2 for (real model + EnKF)'))
if ans == 1:
    csvFileName = 'nominal model.csv'
    std_noise_Q = 0.00000000000000000000000000000000000000000000000000000000000000001
    std_noise_R = 0.00000000000000000000000000000000000000000000000000000000000000001

elif ans == 2:
    csvFileName = 'real model.csv'
    # User selection: Large Q/R (process noise/measurement noise) or Large R/Q?
    ans = int(input('Type 1 for large process noise (small measurement noise), or 2 '
                    'for small process noise (large measurement noise)'))
    if ans == 1:
        Q_to_R = 100
        std_noise_R = 0.001  # measurement noise
        std_noise_Q = std_noise_R * Q_to_R  # process noise
    elif ans == 2:
        R_to_Q = 100
        std_noise_Q = 0.001  # process noise
        std_noise_R = std_noise_Q * R_to_Q  # measurements noise
    else:
        print('Invalid input')
else:
    print('Invalid input')
    
# Parameters of the system
timeSpan = 48  # time span of simulation

# sigma points
N = 10

# Number of measurements
dim_z = 1

blockRun=BasicRunner(argv=["blockMesh"],
                     silent=True,
                     server=False)
print ("Running blockMesh")

blockRun.start()
if not blockRun.runOK():
    error("There was a problem with blockMesh")
print ("Running")

for x in range(0, 1):
        control = ParsedParameterFile('system/controlDict')
        control["startTime"]=x*86400  
        control["endTime"]=(x+1)*86400
        print("start time is" , control["startTime"])
        control.writeFile()
        
        theRun=BasicRunner(argv=["RichardsFoam2_PF"])
        theRun.start()
        
        thetaList = []
        i = False
        Theta = open('86400//theta', 'r')
        csvReader = csv.reader(Theta)
        for line in Theta:
            if line == '(\n':
                i = True
            if i == True:
                thetaList.append(line)
            if line == ')\n':
                i == False
                break