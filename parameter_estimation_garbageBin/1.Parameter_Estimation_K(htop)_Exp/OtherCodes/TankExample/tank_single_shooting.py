# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 08:21:46 2017

@author: Benjamin
"""
import numpy as np
from casadi import *

# Model parameters
r1 = 0.087
r2 = 0.057
hmax = 0.3
A = (np.pi*r1**2) - (np.pi*r2**2)
k = 4.0107*10**-5
umax = k*np.sqrt(hmax)
# Steady state values
x_s = 0.141
u_s = 1.5*10**-5

# Simulation parameters
N = 200

# ODE 
def getODE(x, u):
    qin = u
    return (qin - k*np.sqrt(x))/A

# Create symbolic variables
x = SX.sym('x')
u = SX.sym('u')

f_x = getODE(x, u)

f_q = (x-x_s)*(x-x_s).T + (u-u_s)*(u-u_s).T

# Create integrator
ode = {'x':x, 'p':u, 'ode':f_x, 'quad':f_q}
opts = {'tf':0.5}
I = integrator('I', 'cvodes', ode, opts)

# All control inputs
U = MX.sym('U',N)

# Construct graph of integrator calls
G  = np.array([0.05])
J = 0
for k in range(N):
  Ik = I(x0=G, p=U[k])
  G = Ik['xf']
  J += Ik['qf']
  
 # Allocate an NLP solver
nlp = {'x':U, 'f':J, 'g':G}
opts = {"ipopt.linear_solver":"ma97"}
solver = nlpsol("solver", "ipopt", nlp, opts)

sol = solver(lbx = 0.0, # Lower variable bound
             ubx =  umax,  # Upper variable bound
             lbg = 0,
             ubg = hmax,
             x0 = x_s
             )
print sol

