# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:43:31 2018

@author: Benjamin Decardi-Nelson (decardin@ualberta.ca)

This is a CSTR parameter estimation example using orthogonal collocation on finite element.
dxdt = [
        F0*(c0 - c)/(np.pi*r**2*h) - rate,
        F0*(T0 - T)/(np.pi*r**2*h)
                    - dH/(rho*Cp)*rate
                    + 2*UA/(r*rho*Cp)*(Tc - T),
        (F0 - F)/(np.pi*r**2)
            ]

rate - k0*c*np.exp(-E/T) = g(x,z,u,p)

The parameter to be estimated is k0

"""

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import sys

# ---------------------------------------------------------------------
# Parameter estimation configuration
# ---------------------------------------------------------------------
# Estimation parameters
tf = 10.0                                        # End time
nk = 100                                          # Number of data points
h = tf/nk                                        # Size of the finite elements
c = 3                                            # Number of collocation points (start and end points included)
d = c-1                                          # Degree of interpolating polynomial
cost_type = 1                               # 0 = SSE (sum of squared error) or 1 = ASSE (Average of SSE)
use_jit = 0                                 # 0 for No, 1 for Yes. Use JIT (Just-in-time) compiler for added speed. For small problems JIT does not make a difference.

# Bounds on variables
# Differential state bounds and initial guess
x_min =  np.array([0.0000001,293.15,0.000001])         # Lower bound on states
x_max =  np.array([1.0,400.5,1.0])       # Upper bound on states
xi_min = np.array([0.878,324.5,0.659])      # Initial condition/state [1]
xi_max = np.array([0.878,324.5,0.659])     # Initial condition/state. Same value as [1]
x_init = np.array([0.878,324.5,0.659])       # Initial state guess

# Algebraic state bounds and initial guess
z_min =  np.array([-inf])           # Lower bound on states
z_max =  np.array([inf])            # Upper bound on states
z_init = np.array([0.123])          # Initial state guess

# Parameters and initial guess
p_min = np.array([7.2E10])       # Parameter's lower bound
p_max =  np.array([7.2E10])          # Parameter's upper bound
p_init = np.array([3.0E5])       # Initial parameter guess

# Data. Size should match nk
xs = np.array([0.878,324.5,0.659])
us = np.array([300,0.1])

y_exp = np.array(nk*[xs])       # Experimental measurments states/outputs
u_exp = np.array(nk*[us])      # Experimental inputs

# ---------------------------------------------------------------------
# Model information, setup and cost function
# ---------------------------------------------------------------------
# Model parameters

F0 = .1
T0 = 350
c0 = 1
r = .219
k0 = 7.2e10             # True value
E = 8750                # True value
UA = 54.94
rho = 1000
Cp = .239
dH = -5e4

# Dimensions
Nx = 3                                                      # Number of differential states
Nz = 1                                                      # Number of algebraic states
Nu = 2                                                      # Number of inputs
Np = 1                                                      # Number of parameters to estimate
Ny = 3                                                      # Number of outputs

# Declare variables (use scalar graph)
t  = SX.sym("t")                                            # time
u  = SX.sym("u", Nu)                                        # control
x  = SX.sym("x", Nx)                                        # Differential state
z  = SX.sym("z", Nz)                                        # Algebraic state
p = SX.sym("p", Np)                                         # parameters
xdot = SX.sym("xdot", Nx)                                   # xdot

ym = SX.sym("ym", Ny)                                       # measured outputs
yp = SX.sym("yp", Ny)                                       # predicted outputs


# ODE function
def getODE(x, z, u, p):
    
    c = x[0]
    T = x[1]
    h = x[2]
    
    Tc = u[0]
    F = u[1]
    
    rate = z

    k0 = p[0]
   # E = p[1] 
    
    dxdt = [
        F0*(c0 - c)/(np.pi*r**2*h) - rate,
        F0*(T0 - T)/(np.pi*r**2*h)
                    - dH/(rho*Cp)*rate
                    + 2*UA/(r*rho*Cp)*(Tc - T),
        (F0 - F)/(np.pi*r**2)
            ]
    return vertcat(*dxdt)

def getALG(x, z, u, p):
    c = x[0]
    T = x[1]
    h = x[2]
    
    Tc = u[0]
    F = u[1]
    
    rate = z

    k0 = p[0]
    #E = p[1]
    
    res = [rate - k0*c*np.exp(-E/T)]
    return vertcat(*res)

ode = getODE(x, z, u, p)                                     # Symbolic ode
alg = getALG(x, z, u, p)                                     # Symbolic alg

# Formulate system dynamics as implicit function
res_ode = xdot - ode                                        # LHS - RHS
res_alg = alg
res = vertcat(res_ode, res_alg)

ffcn = Function('ffcn', [t,xdot,x,z,u,p],[res])                          # Righthand side of ode

# Stage cost. Write in matrix form if not scalar. Currently in scalar
def stage_cost(yp, ym):
    return mtimes((yp-ym).T,(yp-ym))

cost = stage_cost(yp, ym)
lcost = Function('lcost', [yp, ym], [cost])

# Function to convert states to measured outputs. Do for your system appropriately.
def getOutputs(x, z, u, p):
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
NX = nk*(d+1)*Nx      # Number of collocated differential states
NZ = nk*(d)*Nz      # Number of collocated algebraic states
NU = nk*Nu              # Parametrized controls
NXF = Nx                # Final state
NP = Np
NV = NX+NZ+NU+NXF+NP

# NLP variable vector
V = MX.sym("V",NV)
  
# All variables with bounds and initial guess
vars_lb = np.zeros(NV)
vars_ub = np.zeros(NV)
vars_init = np.zeros(NV)
offset = 0

# Get collocated states and parametrized control
X = np.resize(np.array([],dtype=MX),(nk+1,d+1))
Z = np.resize(np.array([],dtype=MX),(nk,d))
U = np.resize(np.array([],dtype=MX),nk)
P = np.resize(np.array([],dtype=MX),1)

for k in range(nk):  
  # Collocated states
  for j in range(d+1):
    # Get the expression for the state vector
    X[k,j] = V[offset:offset+Nx]
    
    if j != 0:
        Z[k,j-1] = V[offset+Nx:offset+Nx+Nz]
        
    #  # Add the initial condition and bounds
    if k==0 and j==0:
        vars_init[offset:offset+Nx] = x_init
        vars_lb[offset:offset+Nx] = xi_min
        vars_ub[offset:offset+Nx] = xi_max
        offset += Nx
    else:
        if j != 0:
            vars_init[offset:offset+Nx+Nz] = np.append(x_init, z_init)
            vars_lb[offset:offset+Nx+Nz] = np.append(x_min, z_min)
            vars_ub[offset:offset+Nx+Nz] = np.append(x_max, z_max)
            offset += Nx + Nz
        else:
            vars_init[offset:offset+Nx] = x_init
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
    fk = ffcn(T[k,j], xp_jk/2, X[k,j], Z[k,j-1], U[k], P)
    g.append(fk)
    lbg.append(np.zeros(Nx+Nz)) # equality constraints
    ubg.append(np.zeros(Nx+Nz)) # equality constraints

    # Add contribution to objective
  qk = lcost(getOutputs(X[k,0], Z[k,0-1], U[k], P), y_exp[k])
  
  if cost_type == 0:
      J += qk # Use sum not quadrature # F[j]*qk*h
  elif cost_type == 1:
      J += qk*(1.0/nk) # Use sum not quadrature # F[j]*qk*h
  else:
      sys.exit("Invalid cost function type. Should be 0 (SSE) or 1 (ASSE)")

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

# Use just-in-time compilation to speed up the evaluation
if use_jit == 1:
    if Importer.has_plugin('clang'):
      with_jit = True
      compiler = 'clang'
    elif Importer.has_plugin('shell'):
      with_jit = True
      compiler = 'shell'
    else:
      print("WARNING; running without jit. This may result in very slow evaluation times")
      with_jit = False
      compiler = ''
  
# Set optimization options
opts = {}
opts["expand"] = True
opts["ipopt.max_iter"] = 1000
opts["ipopt.linear_solver"] = 'ma57'
# opts["ipopt.tol"] = 1E-30

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
param = res["x"][-Np:]
print(param)

## ----
## RETRIEVE THE SOLUTION
## ----
xD_opt = np.resize(np.array([],dtype=MX),(Nx,(d+1)*(nk)+1))
xA_opt = np.resize(np.array([],dtype=MX),(Nz,(d)*(nk)))
u_opt = np.resize(np.array([],dtype=MX),(Nu,(d+1)*(nk)+1))
offset = 0
offset2 = 0
offset3 = 0
offset4 = 0

for k in range(nk):

    for j in range(d+1):
        xD_opt[:,offset2] = v_opt[offset:offset+Nx][:,0]
        offset2 += 1
        offset += Nx
        if j!=0:
            xA_opt[:,offset4] = v_opt[offset:offset+Nz][:,0]
            offset4 += 1
            offset += Nz
    utemp = v_opt[offset:offset+Nu][:,0]

    for j in range(d+1):
        u_opt[:,offset3] = utemp
        offset3 += 1
    #    u_opt += v_opt[offset:offset+nu]
    offset += Nu

xD_opt[:,-1] = v_opt[offset:offset+Nz][:,0]

tg = np.array(tau_root)*h
for k in range(nk):
    if k == 0:
        tgrid = tg
    else:
        tgrid = np.append(tgrid,tgrid[-1]+tg)
tgrid = np.append(tgrid,tgrid[-1])
# Plot the results
plt.figure(1)
plt.clf()
plt.subplot(2,2,1)
plt.plot(tgrid[:-1],xD_opt[0,:-1],'--')
plt.title("x")
plt.grid
plt.subplot(2,2,2)
plt.plot(tgrid[:-1],xD_opt[1,:-1],'-')
plt.title("y")
plt.grid
plt.subplot(2,2,3)
plt.plot(tgrid[:-1],xD_opt[2,:-1],'-.')
plt.title("w")
plt.grid

plt.figure(2)
plt.clf()
plt.plot(tgrid,u_opt[0,:],'-.')
plt.title("Crane, inputs")
plt.xlabel('time')


# plt.figure(3)
# plt.clf()
# plt.plot(tgrid,xA_plt[0,:],'-.')
# plt.title("Crane, lambda")
# plt.xlabel('time')
# plt.grid()
plt.show()


# The algebraic states are not defined at the first collocation point of the finite elements:
# with the polynomials we compute them at that point
# Da = np.zeros(d)
# for j in range(1,d+1):
#     # Lagrange polynomials for the algebraic states: exclude the first point
#     La = 1
#     for j2 in range(1,d+1):
#         if j2 != j:
#             La *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
#     lafcn = Function('lafcn', [tau], [La])
#     Da[j-1] = lafcn(tau_root[0])
#
# xA_plt = np.resize(np.array([],dtype=MX),(Nz,(d+1)*(nk)+1))
# offset4=0
# offset5=0
# for k in range(nk):
#
#     for j in range(d+1):
#         if j!=0:
#             xA_plt[:,offset5] = xA_opt[:,offset4]
#             offset4 += 1
#             offset5 += 1
#         else:
#             xa0 = 0
#             for j in range(d):
#                 xa0 += Da[j]*xA_opt[:,offset4+j]
#             xA_plt[:,offset5] = xa0
#             #xA_plt[:,offset5] = xA_opt[:,offset4]
#             offset5 += 1
#
# xA_plt[:,-1] = xA_plt[:,-2]

# The results can be retrieved and plotted. I don't know the structure of your system. We can easily do this since we have both the predicted and experimental states.