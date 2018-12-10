# Import all of the basic libraries (you will always need these)
# matplotlib inline
from matplotlib import pyplot as pl
import numpy as np
# Import a library that contains soil moisture properties and functions
import vanGenuchten as vg
import time
# Import ODE solvers
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array
from math import *

start_time = time.time()

# Select which soil properties to use
# p = vg.HygieneSandstone()
p = vg.Loam()

# Richards equation solver
# This is a function that calculated the right hand side of Richards' equation. You
# will not need to modify this function, unless you are doing something advanced.
# This block of code must be executed so that the function can be later called.
def RichardsModel(psi, t, dz, n, p, vg, qTfun, qBot, psiTop, psiBot):
    # Basic properties:
    C = vg.CFun(psi, p)
    # initialize vectors:
    q = np.zeros(n + 1)
    # # Upper boundary
    # if qTop == []:  # Fixed value
    #     KTop = vg.KFun(np.zeros(1) + psiTop, p)
    #     q[n] = -KTop * ((psiTop - psi[n - 1]) / dz * 2 + 1)
    # else:  # Fixed flux
    #     q[n] = qTop

    # if t>10:
    #     q[n] = qTfun(10)
    # else:
    #     q[n] = qTfun(t)
    if t>160920:
        q[n] = qTfun(160920)
    else:
        q[n] = qTfun(t)
    # Lower boundary
    if qBot == []:
        if psiBot == []:
            # Free drainage: fixed flux: fixed gradient
            KBot = vg.KFun(np.zeros(1) + psi[0], p)
            q[0] = -KBot
        else:
            # Type 1 boundary: Fixed gradient
            KBot = vg.KFun(np.zeros(1) + psiBot, p)
            q[0] = -KBot * ((psi[0] - psiBot) / dz * 2 + 1.0)
    else:
        # Type 2 boundary
        q[0] = qBot
    # Internal nodes
    i = np.arange(0, n - 1)
    Knodes = vg.KFun(psi, p)
    Kmid = (Knodes[i + 1] + Knodes[i]) / 2.0

    j = np.arange(1, n)
    q[j] = -Kmid * ((psi[i + 1] - psi[i]) / dz + 1.0)
    if q[n] != 0:
        if q[n]<q[n-1]:  # because here q is a negative value
            print('Water accumulated at the top, time = ', t / 60, 'min(s)')
            KTop = vg.KFun(psiTop, p)
            dhdz = (psiTop - psi[n-1]) * 2 / dz + 1
            q[n] = -KTop*dhdz
    # Continuity
    i = np.arange(0, n)
    dpsidt = (-(q[i + 1] - q[i]) / dz) / C

    return dpsidt


psi = np.linspace(-10,5)
theta = vg.thetaFun(psi,p)
C=vg.CFun(psi,p)
K=vg.KFun(psi,p)

pl.figure()
pl.rcParams['figure.figsize'] = (5.0, 10.0)
pl.subplot(311)
pl.plot(psi,theta)
pl.ylabel(r'$\theta$', fontsize=20)
pl.subplot(312)
pl.plot(psi,C)
pl.ylabel(r'$C$',fontsize=20)
pl.subplot(313)
pl.plot(psi,K)
pl.ylabel(r'$K$', fontsize=20)
pl.xlabel(r'$\psi$', fontsize=20)


# This block of code sets up and runs the model
# Boundary conditions
qTop = []
# qTop = []
qBot=[]
psiTop=0
psiBot=[]

# Grid in space
# dz=0.1
numberOfStates = 32
dz=0.67/numberOfStates
# ProfileDepth=5
ProfileDepth=0.67
z=np.arange(dz/2.0,ProfileDepth,dz)
n=z.size

# Grid in time
# t = np.linspace(0,10,101)
t = np.linspace(0,160920,2683)
qT=np.zeros(len(t))
# qT[10:20] = -0.01
qT[1150:1216]=-(0.004 / (pi * 0.22 * 0.22) / (65 * 60))
qTfun=interp1d(t,qT)

# Initial conditions
# psi0=-z
thetaIni = array([30.2, 8.8, 8.7, 10.0]) / 100  # 2683 case
hIni = zeros(4)
for i in range(0, 4):
    hIni[i] = (((((thetaIni[i] - p['thetaR']) / (p['thetaS'] - p['thetaR'])) ** (
                1. / (-(1 - 1 / (p['n'])))) - 1)) ** (1. / (p['n']))) / (-p['alpha'])

assignPlane = array(
    [8.54925*numberOfStates/32, 7.164*numberOfStates/32, 7.164*numberOfStates/32, 9.12238*numberOfStates/32])  # the sum of assignPlane need to be the same as nodesInZ
section = array([assignPlane[3], (assignPlane[3] + assignPlane[2]),
                (assignPlane[3] + assignPlane[2] + assignPlane[1]),
                 numberOfStates])
psi0 = zeros(numberOfStates)
psi0[0: int(section[0])] = hIni[3]
psi0[int(section[0]):int(section[1])] = hIni[2]
psi0[int(section[1]):int(section[2])] = hIni[1]
psi0[int(section[2]):int(section[3])] = hIni[0]
# Solve
psi=odeint(RichardsModel,psi0,t,args=(dz,n,p,vg,qTfun,qBot,psiTop,psiBot),mxstep=5000000);
print "Model run successfully"
print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))


# Post process model output to get useful information
# Get water content
theta = vg.thetaFun(psi, p)

# Get total profile storage
S = theta.sum(axis=1) * dz

# Get change in storage [dVol]
dS = np.zeros(S.size)
dS[1:] = np.diff(S) / (t[1] - t[0])

# Get infiltration flux
if qTop == []:
    KTop = vg.KFun(np.zeros(1) + psiTop, p)
    qI = -KTop * ((psiTop - psi[:, n - 1]) / dz * 2 + 1)
else:
    qI = np.zeros(t.size) + qTop

# Get discharge flux
if qBot == []:
    if psiBot == []:
        # Free drainage
        KBot = vg.KFun(psi[:, 0], p)
        qD = -KBot
    else:
        # Type 1 boundary
        KBot = vg.KFun(np.zeros(1) + psiBot, p)
        qD = -KBot * ((psi[:, 0] - psiBot) / dz * 2 + 1.0)
else:
    qD = np.zeros(t.size) + qBot


# Plot vertical profiles
pl.figure()
pl.rcParams['figure.figsize'] = (10.0, 10.0)
for i in range(0,t.size-1):
    pl.subplot(121)
    pl.plot(psi[i,:],z)
    pl.subplot(122)
    pl.plot(theta[i,:],z)

pl.subplot(121)
pl.ylabel('Elevation [m]',fontsize=20)
pl.xlabel(r'$\psi$ [m]',fontsize=20)
pl.subplot(122)
pl.xlabel(r'$\theta$ [-]',fontsize=20)


# Plot timeseries
pl.figure()
dt = t[2]-t[1]
pl.plot(t,dS,label='Change in storage')
pl.hold(True)
pl.plot(t,-qI,label='Infiltration')
pl.plot(t,-qD,label='Discharge')
pl.legend(loc="Upper Left")
pl.ylim((0,0.02))

