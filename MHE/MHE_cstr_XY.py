import mpctools as mpc
import casadi
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy import linalg
from numpy import random
from bsm1model import cstr_xy_ode_scale
from bsm1model import measurement_xy_model

random.seed(927) # Seed random number generator.

doPlots = True
fullInformation = False # True for full information estimation, False for MHE.

# Problem parameters.['w']
Nt = 5 # Horizon length
Delta = 0.05 # Time step
Tsim = 20  # Simulation time
Nsim = int(Tsim/Delta) #1345 # Number of data points
tplot = np.arange(Nsim+1)*Delta

Nx = 2   # Number of system states
Nu = 2   # Number of system inputsn

Ny = 1  # Number of system outputs
Nw = Nx  # Number of process disturbances
Nv = Ny  # Number of measurement noise

#sigma_v = np.array([0.03,0.2]) # Standard deviation of the measurements
#sigma_w = np.array([0.002,0.03]) # Standard deviation for the process noise
#sigma_p = 0.5 # Standard deviation for prior
#
## Make covariance matrices.

sigma_v = 0.0025 # Standard deviation of the measurements
sigma_w = 0.005 # Standard deviation for the process noise
sigma_p = .05 # Standard deviation for prior

# Make covariance matrices.
P = np.diag((sigma_p*np.ones((Nx,)))**2) # Covariance for prior.
Q = np.diag((sigma_w*np.ones((Nw,)))**2)
R = np.diag((sigma_v*np.ones((Nv,)))**2)


#P = np.matrix([[0.0004,0],[0,0.09]])
#Q = np.matrix([[0.000004,0],[0,0.0009]])
##R = np.matrix([[0.0009,0],[0,0.04]])
#R = np.matrix([0.09])


# Import data
# data_input = np.loadtxt('contr_signal_cstr.txt')
x0 =  np.array([1,0.2])
x_0 = 1.05*np.array([1,0.2])

# Set the input vector to the cstr
u = np.zeros((Nsim,Nu)) # 2 inputs Tc and F

for kk in range(Nsim):
#    u[kk,0] = data_input[kk,0]
#    u[kk,1] = data_input[kk,1]
    u[kk,0] = 300.0
    u[kk,1] = 0.1
#=================================================================#
# Simulate open-loop system
#=================================================================#

# test if the model works well
#xs = np.array([0.878,324.5,0.659])
#us = np.array([300,0.1])
#derivative =  cstr_xy_ode(xs,us)
#measure = measurement_xy_model(xs)

# Make a simulator.
model_cstr_casadi = mpc.DiscreteSimulator(cstr_xy_ode_scale, Delta, [Nx,Nu,Nw], ["x","u","w"])    

# Convert continuous-time f to explicit discrete-time F with RK4.
F = mpc.getCasadiFunc(cstr_xy_ode_scale,[Nx,Nu,Nw],["x","u","w"],"F",rk4=True,Delta=Delta,M=1)
H = mpc.getCasadiFunc(measurement_xy_model,[Nx],["x"],"H")

# Define stage costs.
def lfunc(w,v):
    return mpc.mtimes(w.T,linalg.inv(Q),w)+mpc.mtimes(v.T,linalg.inv(R),v)
l = mpc.getCasadiFunc(lfunc,[Nw,Nv],["w","v"],"l")
def lxfunc(x):
    return mpc.mtimes(x.T,linalg.inv(P),x)
lx = mpc.getCasadiFunc(lxfunc,[Nx],["x"],"lx")

# First simulate everything.

#w = sigma_w*random.randn(Nsim,Nw)
#
##ww = np.dot(random.randn(Nsim,Nw),np.matrix([[0.002,0],[0,0.03]]))
##w = np.matrix(ww)
#
#vT = np.matrix([np.dot(random.randn(Nsim,Nv),np.array([0.03]))])
#v=vT.T
##sigma_v = np.array([0.03,4.0]) # Standard deviation of the measurements
##sigma_w = np.array([0.02,6.0]) # Standard deviation for the process noise

w = sigma_w*random.randn(Nsim,Nw)
v = sigma_v*random.randn(Nsim,Nv)



usim = u # We use the input vector to the process instead of dummy input
xsim = np.zeros((Nsim+1,Nx))
xsim[0,:] = x0
yclean = np.zeros((Nsim, Ny))
ysim = np.zeros((Nsim, Ny))

# Simulate the process dynamics
for t in range(Nsim):
    yclean[t,:] = measurement_xy_model(xsim[t]) # Get zero-noise measurement.
    ysim[t,:] = yclean[t,:] + v[t,:] # Add noise to measurement.    
    xsim[t+1,:] = model_cstr_casadi.sim(xsim[t,:],usim[t,:],w[t,:])

# Now do estimation.
xhat_ = np.zeros((Nsim+1,Nx))
xhat = np.zeros((Nsim,Nx))
yhat = np.zeros((Nsim,Ny))
vhat = np.zeros((Nsim,Nv))
what = np.zeros((Nsim,Nw))
x0bar = x_0
xhat[0,:] = x0bar
guess = {}

solveroptions = {
            'linear_solver':'mumps' 
            }
        
totaltime = -time.time()
for t in range(Nsim):
    # Define sizes of everything.    
    N = {"x":Nx, "y":Ny, "u":Nu}
    if fullInformation:
        N["t"] = t
        tmin = 0
    else:
        N["t"] = min(t,Nt)
        tmin = max(0,t - Nt)
    tmax = t+1        
    lb = {"x":np.zeros((N["t"] + 1,Nx))}  
    
#    guess = {
#        'x':np.ones((N["t"] + 1,Nx))*x0}
    
    # Call solver. Would be faster to reuse a single solver object, but we
    # can't because the horizon is changing.
    buildtime = -time.time()
    
    solver = mpc.nmhe(f=F, h=H, u=usim[tmin:tmax-1,:],
                      y=ysim[tmin:tmax,:], l=l, N=N, 
                      verbosity=0,
                      lb=lb,guess=guess,Delta=Delta)
    buildtime += time.time()
    solvetime = -time.time()
    sol = mpc.callSolver(solver)
    solvetime += time.time()
    print ("%3d (%5.3g s build, %5.3g s solve): %s"
           % (t, buildtime, solvetime, sol["status"]))
    if sol["status"] != "Solve_Succeeded":
        break
    xhat[t,:] = sol["x"][-1,...] # This is xhat( t  | t )
    yhat[t,:] = measurement_xy_model(xhat[t,:])    
    vhat[t,:] = sol["v"][-1,...]
    if t > 0:
        what[t-1,:] = sol["w"][-1,...]
    
    # Apply model function to get xhat(t+1 | t )
    xhat_[t+1,:] = np.squeeze(F(xhat[t,:], usim[t,:], np.zeros((Nw,))))
    
    # Save stuff to use as a guess. Cycle the guess.
    guess = {}
    for k in set(["x","w","v"]).intersection(sol.keys()):
        guess[k] = sol[k].copy()
    
    # Do some different things if not using full information estimation.    
    if not fullInformation and t + 1 > Nt:
        for k in guess.keys():
            guess[k] = guess[k][1:,...] # Get rid of oldest measurement.
#============================================================================#            
        # Do EKF to update prior covariance, but don't take EKF state. Remove if arrival cost is not needed
        [P, x0bar, _, _] = mpc.ekf(F,H,x=sol["x"][0,...],
            u=usim[tmin,:],w=sol["w"][0,...],y=ysim[tmin,:],P=P,Q=Q,R=R)
        
        # Need to redefine arrival cost.
        def lxfunc(x):
            return mpc.mtimes(x.T,linalg.inv(P),x)
        lx = mpc.getCasadiFunc(lxfunc,[Nx],["x"],"lx")
#============================================================================#         
     # Add final guess state for new time point.
    for k in guess.keys():
        guess[k] = np.concatenate((guess[k],guess[k][-1:,...]))

totaltime += time.time()
print "Simulation took %.5g s." % totaltime


x_actual_hat = np.zeros((Nsim,Nx))
x_actual = np.zeros((Nsim,Nx))

# scaling parameters for the ODE -- x = x_scale * delta + x_min

delta = np.array([0.1226,14.7143])
minvalue = np.array([0.8774,310])
#delta_1 = 0.1226
#min_1 = 0.8774
#delta_2 = 14.7143
#min_2 = 310
# Recover the actual states and estimates based on the scaled states and estimates
for i in range(Nx):
    x_actual_hat [:,i] = xhat [:,i] * delta[i] + minvalue[i]
    x_actual [:,i] = xsim [:400,i] * delta[i] + minvalue[i]   

np.savetxt("x_from_py.txt",x_actual)      # this saves the data of x
np.savetxt("estimate_from_mhe.txt",x_actual_hat)      # this saves the data of the MHE estimate

# Plots.
if doPlots:
    [fig, ax] = plt.subplots(nrows=2)
    xax = ax[0]
    x2ax = ax[1]

    # Plot state (Concentration)
    colors = ["red","blue"]
    species = ["Concentration", "Temp"]    
    for (i, (c, s)) in enumerate(zip(colors, species)):
        xax.plot(tplot[:-1], x_actual[:,0], color=c, label="$c_%s$" % s)
        xax.plot(tplot[:-1], x_actual_hat[:,0], marker="o", color=c, markersize=3, 
             markeredgecolor=c, linestyle="", label=r"$\hat{c}_%s$" % s)
    mpc.plots.zoomaxis(xax, xscale=1.05, yscale=1.05)
    xax.set_ylabel("Concentration")
    xax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    
        # Plot state (Temperature)   
    colors = ["red","blue"]
    species = ["Concentration", "Temp"]    
    for (i, (c, s)) in enumerate(zip(colors, species)):
        x2ax.plot(tplot[:-1], x_actual[:,1], color=c, label="$c_%s$" % s)
        x2ax.plot(tplot[:-1], x_actual_hat[:,1], marker="o", color=c, markersize=3, 
             markeredgecolor=c, linestyle="", label=r"$\hat{c}_%s$" % s)
    mpc.plots.zoomaxis(xax, xscale=1.05, yscale=1.05)
    xax.set_ylabel("Concentration")
    xax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    
#    # Plot measurements.
#    yax.plot(tplot[:-1], yclean[:,0], color="black", label="$P$")
#    yax.plot(tplot[:-1], yhat[:,0], marker="o", markersize=3, linestyle="",
#             markeredgecolor="black", markerfacecolor="black",
#             label=r"$\hat{P}$")
#    yax.plot(tplot[:-1], ysim[:,0], color="gray", label="$P + v$")
#    yax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
#    yax.set_ylabel("Pressure")
#    yax.set_xlabel("Time")

    # Tweak layout and save.
    fig.subplots_adjust(left=0.1, right=0.8)
    mpc.plots.showandsave(fig,"nmheexample.pdf")
