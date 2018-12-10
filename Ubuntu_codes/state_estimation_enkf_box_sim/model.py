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
import math
import csv
from numpy.random import randn
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import inv
from numpy import dot, zeros, eye, outer
from numpy.random import multivariate_normal

# User selection: Nominal model, or real model?
ans = int(input('Type 1 for nominal model, or 2 for real model'))
if ans == 1:
    processNoise = 0
    csvFileName = 'nominal model.csv'
elif ans == 2:
    processNoise = 0.05
    csvFileName = 'real model.csv'
else:
    print('Invalid input')

# Steady state
#h1_ss = 1.5  # steady state of water level in tank 1 [m]
#h2_ss = 2.0  # steady state of water level in tank 2 [m]
#qi_ss = 0.75  # steady state of inlet water flow rate [m^3/hr]

# Parameters of the system
#A = 4  # cross-section area of bottom of the tank [m^2]
#R1 = 2  # valve 1 resistance [hr/m^3]
#R2 = 3  # valve 2 resistance [hr/m^3]
#cv1 = (2*math.sqrt(h1_ss))/R1
#cv2 = (2*math.sqrt(h2_ss))/R2
dt = 1  # time step [hr]
timeSpan = 48

# Initial condition
#qi_i = [1]  # initial inlet water flow rate [m^3/hr]
#h_i = [0.5, 0.5]  # initial state

# Variance / Covariance
stateCov = 0  # State covariance
std_noise = 0.0  # Process and measurement covariance

# sigma points
N = 10

# Number of measurements
dim_z = 1

# h function, which maps state matrix into predicted measurements
#H = np.array([0, 1])


#def hx(x):

#    return np.dot(H, x)


# fx function, which is the mathematical model of the process
#def fx(x, u):
#    x1 = x[0]
#    x2 = x[1]
#    if x1 < 0:  # Water levels in both tanks cannot be negative
#        x1 = 0
#    if x2 < 0:
#        x2 = 0
#    x1_p = ((-dt*cv1)/A)*math.sqrt(x1) + x1 + (dt/A)*u + randn()*processNoise
#    x2_p = ((dt*cv1)/A)*math.sqrt(x1) - ((dt*cv2)/A)*math.sqrt(x2) + x2 + randn()*processNoise
#    return [x1_p, x2_p]


# Define EnKF
#x = np.array(h_i)
#u = np.array(qi_i)
#P = np.eye(2) * stateCov
f = EnKF(x=x, u=u, P=P, dim_z=dim_z, dt=dt, N=N,
         hx=hx, fx=fx)

f.R *= std_noise ** 2
f.Q *= std_noise ** 2

# Generating lists for data collection and for plots
prediction1 = [x[0]]
prediction2 = [x[1]]

# Prediction using the model
for i in range(timeSpan):
    x_p = f.predict()
    prediction1.append(x_p[0])
    prediction2.append(x_p[1])

time = []
for i in range(timeSpan+1):
    time.append(i)

# Write to csv file
with open(csvFileName, 'w') as csvFile:
    table = csv.writer(csvFile)
    table.writerow(time)
    table.writerow(prediction1)
    table.writerow(prediction2)

#Plot the graph
plt.plot(time, prediction1, label='Predicted water level in tank 1')
plt.plot(time, prediction2, label='Predicted water level in tank 2')
plt.xlabel('Time, t (hr)')
plt.ylabel('Predicted water level, h (m)')
plt.title('Predicted water level as a function of time')
plt.legend()
plt.show()