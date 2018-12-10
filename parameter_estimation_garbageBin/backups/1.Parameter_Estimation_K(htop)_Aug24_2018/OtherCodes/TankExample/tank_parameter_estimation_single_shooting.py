# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:14:04 2018

@author: Benjamin
"""

import numpy as np
from casadi import *

# Model parameters
r1 = 0.087
r2 = 0.057
hmax = 0.3
A = (np.pi*r1**2) - (np.pi*r2**2)
k = SX.sym('k') # Valve coefficient to be estimated. Used as a parameter in the ode # 4.0107*10**-5           
umax = k*np.sqrt(hmax)

# Steady state values
x_s = 0.141
u_s = 1.5*10**-5

# Simulation parameters
N = 200

# Create symbolic variables
x = SX.sym('x')     # State
u = SX.sym('u')     # Input

# ODE function
def getODE(x, u, p):
    qin = u
    return (qin - p*np.sqrt(x))/A

f_x = getODE(x, u, k)          # Symbolic ode

f_q = (x-x_s)*(x-x_s).T     # Quadratic cost function

# Create integrator
ode = {'x':x, 'p':vertcat(u,k), 'ode':f_x, 'quad':f_q}     # Cost is computed using quadrature (numerical integration)
opts = {'tf':0.5}       # step size 0.5 s
I = integrator('I', 'cvodes', ode, opts)        # Build casadi integrator

# All control inputs
U = u_s         # Input is constant for this example. Change to your input sequence

# Parameters
K = MX.sym('K')     # Create an MX symbolic parameters for the Integration

# Construct graph of integrator calls
G  = x_s # Initial state. For this example it is fixed at the steady state value. Set this to the first value of data
J = 0               # Initialize cost function
for i in range(N):
  Ik = I(x0=G, p=vertcat(U,K))
  G = Ik['xf']
  J += Ik['qf']
  
 # Allocate an NLP solver
nlp = {'x':K, 'f':J, 'g':G}
opts = {"ipopt.linear_solver":"mumps"}
solver = nlpsol("solver", "ipopt", nlp, opts)

sol = solver(lbx = -inf, # Lower bound on decision variable
             ubx =  inf,  # Upper bound on decision variable
             lbg = 0,   # lower bound on the states
             ubg = hmax,    # upper bound on the states
             x0 = 1.0107*10**-5 # x_s    # Initial guess of decision variable
             )
print sol
p_est = sol["x"].full().squeeze()
print ""
print("Estimated parameter(s) is(are): " + str(p_est))
print ""
print "Actual value(s) is(are): " + str(4.0107*10**-5)
