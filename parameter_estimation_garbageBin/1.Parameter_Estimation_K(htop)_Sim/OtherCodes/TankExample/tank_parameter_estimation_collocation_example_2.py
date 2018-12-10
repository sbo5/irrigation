# -*- coding: utf-8 -*-
"""
Created on Mon Aug 06 19:43:31 2018

@author: Benjamin
"""

from casadi import *
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Parameter estimation configuration
# ---------------------------------------------------------------------
# Estimation parameters
tf = 10.0                                        # End time
nk = 50                                          # Number of data points
h = tf/nk                                        # Size of the finite elements
c = 4                                            # Number of collocation points (start and end points included)
d = c-1                                          # Degree of interpolating polynomial

# Bounds on variables
# State bounds and initial guess
x_min = np.array([0])          # Lower bound on states
x_max = np.array([0.3])       # Upper bound on states
xi_min = np.array([0.141])      # Initial condition/state [1]
xi_max = np.array([0.141])      # Initial condition/state. Same value as [1]
x_init = np.array([0.01])       # Initial state guess

# Parameters and initial guess
p_min = np.array([1E-30])       # Parameter's lower bound
p_max = np.array([1])          # Parameter's upper bound
p_init = np.array([1E-5])       # Initial parameter guess

# Data. Size sould match nk
y_exp = np.ones(nk)*0.141       # Experimental measurments states/outputs
u_exp = np.ones(nk)*1.5*10**-5  # Experimental inputs

# ---------------------------------------------------------------------
# Model information, setup and cost function
# ---------------------------------------------------------------------
# Model parameters
r1 = 0.087
r2 = 0.057
hmax = 0.3
A = (np.pi*r1**2) - (np.pi*r2**2)                                            # Valve coefficient to be estimated. Used as a parameter in the ode # 4.0107*10**-5           
umax = 4.0107*10**-5 *np.sqrt(hmax)

# Dimensions
Nx = 1                                                      # Number of differential states
Nu = 1                                                      # Number of inputs
Np = 1                                                      # Number of parameters to estimate
Ny = 1                                                      # Number of outputs

# Declare variables (use scalar graph)
t  = SX.sym("t")                                            # time
u  = SX.sym("u", Nu)                                        # control
x  = SX.sym("x", Nx)                                        # state
p = SX.sym("p", Np)                                         # parameters

ym = SX.sym("ym", Ny)                                       # measured outputs
yp = SX.sym("yp", Ny)                                       # predicted outputs


# ODE function
def getODE(x, u, p):
    qin = u
    return (qin - p*np.sqrt(x))/A

xdot = getODE(x, u, p)                                       # Symbolic ode
f = Function('f', [t,x,u,p],[xdot])                          # Righthand side of ode

# Stage cost. Write in matrix form if not scalar. Currently in scalar
def stage_cost(yp, ym):
    return (yp-ym)**2

cost = stage_cost(yp, ym)
lcost = Function('lcost', [yp, ym], [cost])

# Function to convert states to measured outputs. Do for your system appropriately.
def getOutputs(x, u, p):
    return x

# ---------------------------------------------------------------------
# NLP forumation, constraints, bounds and collocation setup. Do not touch anything here unless you know what you are doing. Consult decardin@ualberta.ca
# ---------------------------------------------------------------------

tau_root = [0] + collocation_points(d, "radau")             # Choose collocation points
C = np.zeros((d+1,d+1))                                     # Coefficients of the collocation equation
D = np.zeros(d+1)                                           # Coefficients of the continuity equation

# Coefficients of the quadrature function
# F = np.zeros(d+1)

# Construct polynomial basis
for j in range(d+1):
  # Construct Lagrange polynomials to get the polynomial basis at the collocation point
  pol = np.poly1d([1])
  for r in range(d+1):
    if r != j:
      pol *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
  
  # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
  D[j] = pol(1.0)

  # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
  pder = np.polyder(pol)
  for r in range(d+1):
    C[j,r] = pder(tau_root[r])

  # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
#  pint = np.polyint(pol)
#  F[j] = pint(1.0)

# All collocation time points
T = np.zeros((nk,d+1))
for k in range(nk):
  for j in range(d+1):
    T[k,j] = h*(k + tau_root[j])

    
# Control bounds
u_min = u_exp
u_max = u_exp
u_init = u_exp

# Total number of variables
NX = nk*(d+1)*Nx      # Collocated states
NU = nk*Nu              # Parametrized controls
NXF = Nx                # Final state
NP = Np
NV = NX+NU+NXF+NP

# NLP variable vector
V = MX.sym("V",NV)
  
# All variables with bounds and initial guess
vars_lb = np.zeros(NV)
vars_ub = np.zeros(NV)
vars_init = np.zeros(NV)
offset = 0

# Get collocated states and parametrized control
X = np.resize(np.array([],dtype=MX),(nk+1,d+1))
U = np.resize(np.array([],dtype=MX),nk)
P = np.resize(np.array([],dtype=MX),1)
for k in range(nk):  
  # Collocated states
  for j in range(d+1):
    # Get the expression for the state vector
    X[k,j] = V[offset:offset+Nx]
    
    # Add the initial condition
    vars_init[offset:offset+Nx] = x_init
    
    # Add bounds
    if k==0 and j==0:
      vars_lb[offset:offset+Nx] = xi_min
      vars_ub[offset:offset+Nx] = xi_max
    else:
      vars_lb[offset:offset+Nx] = x_min
      vars_ub[offset:offset+Nx] = x_max
    offset += Nx
  
  # Parametrized controls
  U[k] = V[offset:offset+Nu]
  vars_lb[offset:offset+Nu] = u_min[k]
  vars_ub[offset:offset+Nu] = u_max[k]
  vars_init[offset:offset+Nu] = u_init[k]
  offset += Nu
  
# State at end time
X[nk,0] = V[offset:offset+Nx]
vars_lb[offset:offset+Nx] = x_min
vars_ub[offset:offset+Nx] = x_max
vars_init[offset:offset+Nx] = x_init
offset += Nx

# Parameter
P = V[offset:offset+Np]
vars_lb[offset:offset+Np] = p_min
vars_ub[offset:offset+Np] = p_max
vars_init[offset:offset+Np] = p_init
offset += Np

assert (offset == NV)

  
# Constraint function for the NLP
g = []
lbg = []
ubg = []

# Objective function
J = 0

# For all finite elements
for k in range(nk):
  
  # For all collocation points
  for j in range(1,d+1):
        
    # Get an expression for the state derivative at the collocation point
    xp_jk = 0
    for r in range (d+1):
      xp_jk += C[r,j]*X[k,r]
      
    # Add collocation equations to the NLP
    fk = f(T[k,j], X[k,j], U[k], P)
    g.append(h*fk - xp_jk)
    lbg.append(np.zeros(Nx)) # equality constraints
    ubg.append(np.zeros(Nx)) # equality constraints

    # Add contribution to objective
  qk = lcost(getOutputs(X[k,0], U[k], P), y_exp[k])
  J += qk # Use sum not quadrature # F[j]*qk*h

  # Get an expression for the state at the end of the finite element
  xf_k = 0
  for r in range(d+1):
    xf_k += D[r]*X[k,r]

  # Add continuity equation to NLP
  g.append(X[k+1,0] - xf_k)
  lbg.append(np.zeros(Nx))
  ubg.append(np.zeros(Nx))
  
# Concatenate constraints
g = vertcat(*g)
  
# NLP
nlp = {'x':V, 'f':J, 'g':g}

## ----
## SOLVE THE NLP
## ----

# Set optimization options
opts = {}
opts["expand"] = True
#opts["ipopt.max_iter"] = 4
opts["ipopt.linear_solver"] = 'ma57'

# Allocate an NLP solver
solver = nlpsol("solver", "ipopt", nlp, opts)
arg = {}
  
# Initial condition
arg["x0"] = vars_init

# Bounds on x
arg["lbx"] = vars_lb
arg["ubx"] = vars_ub

# Bounds on g
arg["lbg"] = np.concatenate(lbg)
arg["ubg"] = np.concatenate(ubg)

# Solve the problem
res = solver(**arg)

# ---------------------------------------------------------------------
# Solution presentation and plots
# ---------------------------------------------------------------------

print
print("=======================================")
# Print the optimal cost
print("optimal cost: ", float(res["f"]))

# Retrieve the solution
v_opt = np.array(res["x"])
print("The estimated parameters are: ")
param = res["x"][-Np]
print(param)

# The results can be retrieved and plotted. I don't know the structure of your system. We can easily do this since we have both the predicted and experimental states.