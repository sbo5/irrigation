# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:08:20 2017

@author: X. Yin
"""
from __future__ import division
import numpy as np
import scipy.io as spio



# measurement functions of bsm1
def measurement_bsm1(x):
#    mat_measure = spio.loadmat('C_145.mat', squeeze_me=True)
#    C_matrix = mat_measure['C_145']
    C_matrix = np.identity(145)
    y_145 = np.dot(C_matrix,x)
    
    """Pressure measurement."""
    return np.array(y_145)


#==========================================#
# Ben simulation model   
# Define model parameters
F0 = .1
T0 = 350
c0 = 1
r = .219
k0 = 7.2e10
E = 8750
U = 54.94
rho = 1000
Cp = .239
dH = -5e4

# Define steady state values
xs = np.array([0.878,324.5,0.659])
us = np.array([300,0.1])

# Create CSTR ode model
def cstr_xy_ode(x, u, w=np.zeros(2)):
    
    c = x[0]
    T = x[1]
    h = 0.659 
    Tc = u[0]
    F = u[1]
    
    rate = k0*c*np.exp(-E/T)
    
    dxdt = [
        F0*(c0 - c)/(np.pi*r**2*h) - rate + w[0],
        F0*(T0 - T)/(np.pi*r**2*h)
                    - dH/(rho*Cp)*rate
                    + 2*U/(r*rho*Cp)*(Tc - T) + w[1]
            ]
    return np.array(dxdt)
#==========================================#
def cstr_xy_ode_scale(x_scale, u, w=np.zeros(2)):
    
# scaling parameters for the ODE -- x = x_scale * delta + x_min    
    delta_1 = 0.1226
    min_1 = 0.8774
    delta_2 = 14.7143
    min_2 = 310
    
    c = x_scale[0] * delta_1 + min_1       # c = x[0]
    T = x_scale[1] * delta_2 + min_2       # T = x[1]
    
    h = 0.659 
    Tc = u[0]
    F = u[1]
    
    rate = k0*c*np.exp(-E/T)
    
    dx_scaledt = [
        1/delta_1*(F0*(c0 - c)/(np.pi*r**2*h) - rate + w[0]),
        1/delta_2*(F0*(T0 - T)/(np.pi*r**2*h)
                    - dH/(rho*Cp)*rate
                    + 2*U/(r*rho*Cp)*(Tc - T) + w[1])
            ]

    return np.array(dx_scaledt)
#==========================================#
# measurement functions of Ben's model
def measurement_xy_model(x):
    # mat_measure = spio.loadmat('C_145.mat', squeeze_me=True)
###    C_matrix = mat_measure['C_145']
#    C_matrix_ben = np.identity(2)
#    y_3_ben = np.dot(C_matrix_ben,x)
#    return np.array(y_3_ben)
    
    """Pressure measurement."""
#
    C_matrix_ben = np.matrix([1,2])   
#    C_matrix_ben = np.matrix([1,2])  # use this for illustration only  
    y_3_ben = np.dot(C_matrix_ben,x)
    return y_3_ben



