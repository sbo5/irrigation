from __future__ import (absolute_import, print_function, division, unicode_literals)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
from casadi import *
from math import *
import matplotlib.pyplot as plt

print("I am starting up")
start_time = time.time()
# First sections: Geometry/Physical domain#######################################################################
# Define the geometry
lengthOfZ = 0.67  # meter
ratio_z = 1
nodesInZ = int(32/ratio_z)
nodesInPlane = 1
numberOfNodes = nodesInZ
dzList = lengthOfZ/nodesInZ

# Second sections: Parameters/Initial conditions ######################################################################
# Initial Parameters
Ks = 1.  # [m/s]
Theta_s = 1.
Theta_r = 1.
Alpha = 1.
N = 1.

Ks_s = 4.01989044e-06  # [m/s]
Theta_s_s = 4.00000000e-01
Theta_r_s = 7.56684383e-02
Alpha_s = 2.47860000e+00
N_s = 1.50928864e+00

# # Loam
# Ks_s = 0.00000288889  # [m/s]
# Theta_s_s = 0.43
# Theta_r_s = 0.078
# Alpha_s = 3.6
# N_s = 1.56

# # Loamy sand
# Ks_s = 14.59/100/3600  # [m/s]
# Theta_s_s = 0.41
# Theta_r_s = 0.057
# Alpha_s = 0.124*100
# N_s = 2.28

# # This is the results obtained from 30% bounds for initial guess (Loam), 2683
# Ks_s = 3.32222350e-06  # [m/s]
# Theta_s_s = 3.97938207e-01
# Theta_r_s = 7.93338514e-02
# Alpha_s = 3.06000001e+00
# N_s = 1.61664621e+00

S = 0.0001  # [per m]
PET = 0.0000000070042  # [per sec]
mini = 1e-20
thetaIni = array([30.2, 8.8, 8.7, 10.0]) / 100
# thetaIni = array([10.1, 8.4, 8.6, 10.0]) / 100


# Calculate the initial state
def ini_state(p):
    # Initial state
    hIni = ((((((thetaIni - p[2] * Theta_r_s) / (p[1] * Theta_s_s - p[2] * Theta_r_s + mini)) + mini) ** (
                1. / (-(1 - 1 / (p[4] * N_s + mini)) + mini)) - 1) + mini) ** (1. / (p[4] * N_s + mini))) / (
                          -p[3] * Alpha_s + mini)

    assignPlane = array([8.54925, 7.164, 7.164, 9.12238])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    section = array([nodesInPlane*assignPlane[3], nodesInPlane*(assignPlane[3]+assignPlane[2]),
                    nodesInPlane*(assignPlane[3]+assignPlane[2]+assignPlane[1]),
                    numberOfNodes])
    # hMatrix = MX.zeros(numberOfNodes)
    hMatrix = np.zeros(numberOfNodes)
    hMatrix[0: int(section[0])] = hIni[3]
    hMatrix[int(section[0]):int(section[1])] = hIni[2]
    hMatrix[int(section[1]):int(section[2])] = hIni[1]
    hMatrix[int(section[2]):int(section[3])] = hIni[0]
    return hMatrix, hIni


# Third sections: ODEs #################################################################################################
def thetaFun(h, p):
    ks = p[0]
    theta_s = p[1]
    theta_r = p[2]
    alpha = p[3]
    n = p[4]
    theta = 0.5*(1-sign(h))*(100*((theta_s*Theta_s_s-theta_r*Theta_r_s)*(1+(-alpha*Alpha_s*(-1)*(h**2)**(1./2.))**(n*N_s))**(-(1-1/(n*N_s)))+theta_r*Theta_r_s))\
            +0.5*(1+sign(h))*theta_s*Theta_s_s*100
    return theta


# Calculation of hydraulic conductivity
def hydraulic_conductivity(h, p):
    term3 = ((1+(((-1*(p[3]*Alpha_s)*-1*(h**2+mini)**(1./2.))+mini)**(p[4]*N_s)))+mini)
    term4 = ((term3**(-(1-1/(p[4]*N_s+mini))))+mini)
    term5 = term4**(1./2.)
    term6 = term4**((p[4]*N_s)/((p[4]*N_s)-1+mini))
    term7 = (1-term6+mini)**(1-1/(p[4]*N_s+mini))
    term8 = ((1-term7)**2)
    term1 = ((1 + sign(h)) * (p[0]*Ks_s))
    term2 = (1-sign(h))*(p[0]*Ks_s)*term5*term8
    term0 = (term1+term2)
    hc = 0.5*term0
    return hc


# Calculation of capillary capacity
def capillary_capacity(h, p, s=S):
    cc = 0.5*(((1+np.sign(h))*s)+
              (1-np.sign(h))*(s+((p[1]*Theta_s_s-p[2]*Theta_r_s)*(p[3]*Alpha_s)*(p[4]*N_s)*(1-1/(p[4]*N_s+mini)))*((-1*(p[3]*Alpha_s)*-1*((h)**2)**(0.5))**((p[4]*N_s)-1))*
                              (((1+((-1*(p[3]*Alpha_s)*-1*((h)**2+mini)**(0.5))+mini)**(p[4]*N_s))+mini)**(-(2-1/(p[4]*N_s+mini))))))
    return cc


def mean_hydra_conductivity(left_boundary, right_boundary, p):
    lk = hydraulic_conductivity(left_boundary, p)
    rk = hydraulic_conductivity(right_boundary, p)
    mk = mean([lk, rk])
    return mk


# def qpet(pet=PET):
#     q_pet = pet*()
# def aet():


def RichardsEQN_1D(x, u, p):
    # Optimized parameters
    irr = u
    state = x
    dhdt = SX.zeros(numberOfNodes)
    # dhdt = zeros(numberOfNodes)
    dz = dzList
    for i in range(0, numberOfNodes):
        current_state = state[i]
        if i == 0:
            bc_zl = current_state
            bc_zu = state[i + nodesInPlane]

            # KzL = hydraulic_conductivity(bc_zl, p)
            KzL = mean_hydra_conductivity(bc_zl, state[i], p)
            KzU = mean_hydra_conductivity(state[i], bc_zu, p)
            deltaHzL = (state[i] - bc_zl) / dz
            deltaHzU = (bc_zu - state[i]) / dz
            temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
            temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
            temp4 = 0  # source term
        elif i == nodesInZ - 1:
            bc_zl = state[i - nodesInPlane]
            KzL = mean_hydra_conductivity(state[i], bc_zl, p)
            deltaHzL = (state[i] - bc_zl) / dz
            KzU1 = hydraulic_conductivity(current_state, p)
            bc_zu = current_state + dz * (-1 + irr / KzU1)
            bc_zu = sign(bc_zu) * 0.0 + 0.5 * (1 - sign(bc_zu)) * bc_zu
            # if bc_zu == 0:
            #     print('Water accumulated at t = ', t / 60, 'min(s)')
            # KzU = hydraulic_conductivity(bc_zu, p)
            KzU = mean_hydra_conductivity(state[i], bc_zu, p)
            deltaHzU = (bc_zu - state[i]) / dz
            temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
            temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
            temp4 = 0  # source term
        else:
            bc_zl = state[i - nodesInPlane]
            bc_zu = state[i + nodesInPlane]

            KzL = mean_hydra_conductivity(state[i], bc_zl, p)
            KzU = mean_hydra_conductivity(state[i], bc_zu, p)
            deltaHzL = (state[i] - bc_zl) / dz
            deltaHzU = (bc_zu - state[i]) / dz
            temp2 = 1 / (0.5 * 2 * dz) * (KzU * deltaHzU - KzL * deltaHzL)
            temp3 = 1 / (0.5 * 2 * dz) * (KzU - KzL)
            temp4 = 0  # source term
        temp5 = temp2 + temp3 - temp4
        temp6 = temp5 / (capillary_capacity(current_state, p)+mini)
        dhdt[i] = temp6
    return dhdt


def simulate(p):
    h = zeros((len(timeList), numberOfNodes))
    h[0], hIni = ini_state(p)
    h0 = h[0]
    theta = zeros((len(timeList), numberOfNodes))
    h_avg = zeros((len(timeList), 4))  # 4 sensors
    h_avg[0] = hIni
    theta_avg = zeros((len(timeList), 4))
    theta_avg[0] = thetaIni*100
    for i in range(len(timeList)-1):
        print('At time:', i+1, ' min(s)')
        ts = [timeList[i], timeList[i+1]]
        y = integrate.odeint(RichardsEQN_1D, h0, ts, args=(irr_amount[i], p))
        h0 = y[-1]
        h[i+1] = h0

        temp11 = (1 + (-p[3] * h0) ** p[4])
        temp22 = temp11 ** (-(1 - 1 / p[4]))
        temp33 = (p[1] - p[2]) * temp22
        theta0 = 100 * (temp33 + p[2])
        theta[i+1] = theta0
        if ratio_z == 1:
            start = 2
            end = 9
            for j in range(0, 4):
                h_avg[i+1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                theta_avg[i+1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                start += 7
                end += 7
        else:
            start = 2
            end = 5
            for j in range(0, 4):
                h_avg[i+1][j] = (sum(h0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                theta_avg[i+1][j] = (sum(theta0[start * nodesInPlane:end * nodesInPlane]) / (
                        (end - start) * nodesInPlane))
                start += 3
                end += 3

        h_avg[i+1] = h_avg[i+1][::-1]
        theta_avg[i+1] = theta_avg[i+1][::-1]
    return h_avg, theta_avg


def objective(x, p, theta_e):
    # Simulate objective
    ks = p[0]
    theta_s = p[1]
    theta_r = p[2]
    alpha = p[3]
    n = p[4]

    temp11 = ((1 + ((-(alpha*Alpha_s) * x)+mini) ** (n*N_s))+mini)
    temp22 = temp11 ** (-(1 - 1. / (n*N_s+mini)))
    temp33 = (theta_s*Theta_s_s - theta_r*Theta_r_s) * temp22
    theta = 100 * (temp33 + theta_r*Theta_r_s)

    obj = 0
    if ratio_z == 1:
        for i in range(0, 4):
            start = 2
            end = 9
            total = 0
            for j in range(0, 7):
                k = start + i * (end-start)
                total += theta[k]
                start += 1
                end += 1
            theta_avg = total/(end-start)
            # Obj Fcn 1
            obj += (theta_avg - theta_e[-(i+1)])**2
            #
            # Obj fcn 2
            # j = 2 + 7*i + 3
            # obj += (theta[j] - theta_e[-(i+1)])**2

            # # Obj fcn 3
            # temp1 = (((theta_avg/100 - theta_r*Theta_r_s) / ((theta_s*Theta_s_s - theta_r*Theta_r_s)+mini))+mini)
            # temp2 = (temp1 ** (1. / (-(1. - (1. / (n*N_s+mini)))+mini)))
            # temp3 = ((temp2 - 1)+mini)
            # temp4 = (temp3 ** (1. / (n*N_s+mini)))
            # h_avg = temp4 / ((-alpha*Alpha_s)+mini)
            #
            # temp11 = (((theta_e/100 - theta_r*Theta_r_s) / ((theta_s*Theta_s_s - theta_r*Theta_r_s)+mini))+mini)
            # temp22 = (temp11 ** (1. / (-(1. - (1. / (n*N_s+mini)))+mini)))
            # temp33 = ((temp22 - 1)+mini)
            # temp44 = (temp33 ** (1. / (n*N_s+mini)))
            # h_e = temp44 / ((-alpha*Alpha_s)+mini)
            #
            # obj += (h_avg - h_e[-(i + 1)]) ** 2

    else:
        for i in range(0, 4):
            start = 2
            end = 5
            total = 0
            for j in range(0, 3):
                k = start + i * (end-start)
                total += theta[k]
                start += 1
                end += 1
            theta_avg = total/(end-start)
            obj += (theta_avg - theta_e[-(i+1)])**2
    # obj = 0
    return obj


# Fourth section: main##########################################################################################
# Initial decision variables
# Ks = 0.00000288889  # [m/s]
# Theta_s = 0.43
# Theta_r = 0.078
# Alpha = 3.6
# N = 1.56

p0 = array([Ks, Theta_s, Theta_r, Alpha, N])
# h0, hIni = ini_state(p0)
h0 = array([-1.35220202e+01, -1.35220202e+01, -2.51123232e+01, -2.51123232e+01,
 -2.51123232e+01, -2.51123232e+01, -2.41878786e+01, -1.43310908e+01,
 -1.35220202e+01, -7.73809479e+01, -7.73809479e+01, -7.73809479e+01,
 -7.73809479e+01, -7.73809479e+01, -7.73809479e+01, -7.73809479e+01,
 -1.14548957e+02, -1.14548957e+02, -1.25454277e+02, -2.12733778e+02,
 -2.12733778e+02, -2.12733778e+02, -2.12733778e+02, -7.15600795e-01,
 -7.15600795e-01, -7.15600795e-01, -7.15600795e-01, -7.10115029e-01,
 -3.85323497e-01, -3.85323497e-01, -3.08861024e-01, -2.07481883e-01])
plb = 1.01*h0
pub = 0.99*h0

# Time interval
# Irrigation scheduled
ratio_t = 1
ratio_t_farm = 1
dt = 60/ratio_t/ratio_t_farm  # second
# timeSpan = 4000
timeSpan = 2683
# timeSpan = 1444
interval = int(timeSpan*60/dt)

timeList = []
for i in range(interval):
    current_t = i*dt
    timeList.append(current_t)
timeList = array(timeList, dtype='O')

timeList_original = []
for i in range(timeSpan):
    current_t = i*dt*ratio_t
    timeList_original.append(current_t)
timeList_original = array(timeList_original, dtype='O')

# Reading data from the file
he = zeros((timeSpan, 4))  # four sensors
theta_e = zeros((timeSpan, 4))
# with open('Data/exp_data_4L_4000.dat', 'r') as f:
with open('Data/exp_data_4L_2683.dat', 'r') as f:
# with open('Data/exp_data_5L_1444.dat', 'r') as f:
    whole = f.readlines()
    for index, line in enumerate(whole):
        one_line = line.rstrip().split(",")
        one_line = one_line[1:5]
        h_temp = []
        theta_temp = []
        for index1, item in enumerate(one_line):
            item = float(item)
            theta_temp.append(item)
            temp1 = (((item/100 - Theta_r_s) / ((Theta_s_s - Theta_r_s)+mini))+mini)
            temp2 = (temp1 ** (1. / (-(1. - (1. / (N_s+mini)))+mini)))
            temp3 = ((temp2 - 1)+mini)
            temp4 = (temp3 ** (1. / (N_s+mini)))
            item = temp4 / ((-Alpha_s)+mini)
            h_temp.append(item)
        h_temp = array(h_temp, dtype='O')
        theta_temp = array(theta_temp, dtype='O')
        he[index] = h_temp
        theta_e[index] = theta_temp

# Irrigation amount
irr_amount = zeros((interval, 1))
for i in range(0, interval):
    # irr_amount[i] = 0
    # # 1444 case
    # if i in range(0, 22*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    # elif i in range(59*ratio_t, 87*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(161*ratio_t, 189*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(248*ratio_t, 276*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(int(335*ratio_t), int(361*ratio_t)):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (25 * 60))
    # else:
    #     irr_amount[i] = 0
    if i in range(1150*ratio_t, 1216*ratio_t):
        irr_amount[i] = (0.004 / (pi * 0.22 * 0.22) / (65 * 60))
    else:
        irr_amount[i] = 0

# Create symbolic variables
x = SX.sym('x', numberOfNodes, 1)
u = SX.sym('u')
k = SX.sym('k', len(p0), 1)
# xe = SX.sym('xe', 4, 1)  # 4 sensors

# Create ODE and Objective functions
print("I am ready to create symbolic ODE")
f_x = RichardsEQN_1D(x, u, k)
# f_q = objective(x, k, xe)

# Create integrator
print("I am ready to create integrator")
ode = {'x': x, 'p': vertcat(u, k), 'ode': f_x}
opts = {'tf': 60/ratio_t/ratio_t_farm, 'regularity_check':True}  # seconds
# opts = {'tf': 60}
I = integrator('I', 'cvodes', ode, opts)  # Build casadi integrator

# All parameter sets and irrigation amount
G0 = MX.sym('G0', numberOfNodes)
U = irr_amount
K = p0
XE = theta_e
GL = []
# Construct graph of integrator calls
print("I am ready to create construct graph")
J = 0  # Initial cost function
for i in range(interval):
    if i == 0:
        G = G0
    else:
        pass
    Ik = I(x0=G, p=vertcat(U[i], K))  # integrator with initial state G, and input U[k]
    # if i % (ratio_t*10) == 0:
    #     j = int(i/(ratio_t*10))
    #     J += objective(G, K, XE[j])
    if i % (ratio_t) == 0:
        j = int(i/(ratio_t))
        J += objective(G, K, XE[j])
    GL.append(G)
    G = Ik['xf']  # Assign the finial state to the initial state

'''''
# This is used to check if Casadi model return the sam results as mpctool/odeint
K = p0  # Here, K needs to contain real numbers
GL_numpy_array = []
for i in range(interval):
    if i == 0:
        G, hIni = ini_state(K)  # Initial state
    else:
        pass
    Ik = I(x0=G, p=vertcat(U[i], K))  # integrator with initial state G, and input U[k]
    if i % (ratio_t*10) == 0:  # Here, we don actually use the objective function
        j = int(i/(ratio_t*10))
        J += objective(G, K, XE[j])
    if i == 0:
        GL_numpy_array.append(G)
    else:
        G_numpy_array = G.full()  # Convert MX to numpy array
        G_numpy_array = G_numpy_array.ravel()
        GL_numpy_array.append(G_numpy_array)
    G = Ik['xf']  # Assign the finial state to the initial state
GL_numpy_array = array(GL_numpy_array, dtype='O')

# Convert h to theta
temp11 = ((1 + ((-(K[3]*Alpha_s) * GL_numpy_array) + mini) ** (K[4]*N_s)) + mini)
temp22 = temp11 ** (-(1 - 1. / (K[4]*N_s + mini)))
temp33 = (K[1]*Theta_s_s - K[2]*Theta_r_s) * temp22
thetaL_numpy_array = 100 * (temp33 + K[2]*Theta_r_s)

# Take the average
thetaL_numpy_array_avg = zeros((interval, 4))
for index in range(interval):
    for i in range(0, 4):
        start = 2
        end = 9
        total = 0
        for j in range(0, 7):
            k = start + i * (end - start)
            total += thetaL_numpy_array[index][k]
            start += 1
            end += 1
        theta_avg = total / (end - start)
        thetaL_numpy_array_avg[index][-(i+1)] = theta_avg
np.savetxt('sim_results_1D_farm_casadi_2683', thetaL_numpy_array_avg)
'''

print("I am doing creating NLP solver function")
# Allocate an NLP solver
nlp = {'x': G0, 'f': J, 'g': vertcat(*GL)}  # x: Solve for P (parameters), which gives the lowest J (cost fcn), with the constraints G (propagated model)
# nlp = {'x': G0, 'f': J}  # x: Solve for P (parameters), which gives the lowest J (cost fcn), with the constraints G (propagated model)
opts = {"ipopt.linear_solver":"ma97"}
# opts = {"ipopt.linear_solver": "ma57"}
# opts = {"ipopt.linear_solver": "mumps"}
opts["ipopt.hessian_approximation"] = 'limited-memory'
opts["ipopt.print_level"] = 5
opts["regularity_check"] = True
opts["verbose"] = True
# opts["ipopt.tol"]=1e-05
# opts['ipopt.max_iter'] = 30
print("I am ready to build")
solver = nlpsol('solver', 'ipopt', nlp, opts)

print("I am ready to solve")

sol = solver(
        lbx=plb,
        ubx=pub,
        x0=h0  # Initial guess of decision variable
      )
print (sol)
states = sol['x'].full().squeeze()
print ("")
print ("Estimated parameter(s) is(are): " + str(states))
print ("")
print ("Actual value(s) is(are): " + str(h0))

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))
# np.savetxt('sim_results_1D_farm_white_noise', rand_e)
# # io.savemat('sim_results_1D_odeint.mat', dict(y_1D_odeint=theta_i))
