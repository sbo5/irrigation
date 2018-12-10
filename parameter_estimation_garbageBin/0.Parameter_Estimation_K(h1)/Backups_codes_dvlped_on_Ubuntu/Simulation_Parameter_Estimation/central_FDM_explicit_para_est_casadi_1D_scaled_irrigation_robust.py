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
# Define the nodes
ratio_z = 1
nodesInZ = int(32/ratio_z)
# nodesInZinOF = nodesInZ+1
nodesInPlane = 1
numberOfNodes = nodesInZ
dzList = lengthOfZ/nodesInZ
# Label the nodes
indexOfNodes = []
for i in range(0, numberOfNodes):
    indexOfNodes.append(i)
positionOfNodes = []
for k in range(0, nodesInZ):
    positionOfNodes.append([0, 0, k])

# Second sections: Parameters/Initial conditions ######################################################################
# Initial Parameters
Ks = 1.  # [m/s]
Theta_s = 1.
Theta_r = 1.
Alpha = 1.
N = 1.

# Ks_s = 0.00000278889  # [m/s]
# Theta_s_s = 0.44
# Theta_r_s = 0.077
# Alpha_s = 3.5
# N_s = 1.5

Ks_s = 0.00000288889*0.7  # [m/s]
Theta_s_s = 0.43*1.3
Theta_r_s = 0.078*1.3
Alpha_s = 3.6*0.7
N_s = 1.56*1.3

S = 0.0001  # [per m]
PET = 0.0000000070042  # [per sec]
mini = 1e-100
thetaIni = array([30.0, 30.0, 30.0, 30.0]) / 100

# Experimental parameters
Ks_e = 0.00000288889  # [m/s]
Theta_s_e = 0.43
Theta_r_e = 0.078
Alpha_e = 3.6
N_e = 1.56


# Calculate the initial state
def ini_state(p):
    # Initial state
    # hIni = zeros(4)
    # for i in range(0, 4):
        # hIni[i] = ((((((thetaIni[i] - p[2]*Theta_r_s)/(p[1]*Theta_s_s-p[2]*Theta_r_s+mini))+mini)**(1./(-(1-1/(p[4]*N_s+mini))+mini)) - 1)+mini)**(1./(p[4]*N_s+mini)))/(-p[3]*Alpha_s+mini)
    hIni = ((((((thetaIni - p[2] * Theta_r_s) / (p[1] * Theta_s_s - p[2] * Theta_r_s + mini)) + mini) ** (
                1. / (-(1 - 1 / (p[4] * N_s + mini)) + mini)) - 1) + mini) ** (1. / (p[4] * N_s + mini))) / (
                          -p[3] * Alpha_s + mini)

    assignPlane = array([8.54925, 7.164, 7.164, 9.12238])/ratio_z  # the sum of assignPlane need to be the same as nodesInZ
    section = array([nodesInPlane*assignPlane[3], nodesInPlane*(assignPlane[3]+assignPlane[2]),
                    nodesInPlane*(assignPlane[3]+assignPlane[2]+assignPlane[1]),
                    numberOfNodes])
    hMatrix = MX.zeros(numberOfNodes)
    # hMatrix = np.zeros(numberOfNodes)
    hMatrix[0: int(section[0])] = hIni[3]
    hMatrix[int(section[0]):int(section[1])] = hIni[2]
    hMatrix[int(section[1]):int(section[2])] = hIni[1]
    hMatrix[int(section[2]):int(section[3])] = hIni[0]
    return hMatrix, hIni


# Third sections: ODEs #################################################################################################
# Calculation of hydraulic conductivity
def hydraulic_conductivity(h, p):
    # term3 = ((1+(((-1*(alpha*Alpha_s)*-1*(h**2+mini)**(1./2.))+mini)**(n*N_s)))+mini)
    # term4 = ((term3**(-(1-1/(n*N_s+mini))))+mini)
    # term5 = term4**(1./2.)
    # term6 = term4**((n*N_s)/((n*N_s)-1+mini))
    # term7 = (1-term6+mini)**(1-1/(n*N_s+mini))
    # term8 = ((1-term7)**2)
    # term1 = ((1 + sign(h)) * (ks*Ks_s))
    # term2 = (1-sign(h))*(ks*Ks_s)*term5*term8
    # term0 = (term1+term2)
    # hc = 0.5*term0

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

    # hc = (p[0]*Ks_s)*(((1+(((-1*(p[3]*Alpha_s)*h)+mini)**(p[4]*N_s))+mini)**(-(1-1/(p[4]*N_s+mini))))+mini)**(1./2.)*((1-(1-((((1+(-1*(p[3]*Alpha_s)*h+mini)**(p[4]*N_s))+mini)**(-(1-1/(p[4]*N_s+mini))))+mini)**((p[4]*N_s)/((p[4]*N_s)-1+mini))+mini)**(1-1/(p[4]*N_s+mini)))**2)
    return hc


# Calculation of capillary capacity
def capillary_capacity(h, p, s=S):
    # cc = 0.5*(((1+np.sign(h))*s)+
    #           (1-np.sign(h))*(s+((theta_s*Theta_s_s-theta_r*Theta_r_s)*(alpha*Alpha_s)*(n*N_s)*(1-1/(n*N_s+mini)))*((-1*(alpha*Alpha_s)*-1*((h)**2)**(0.5))**((n*N_s)-1))*
    #                           (((1+((-1*(alpha*Alpha_s)*-1*((h)**2+mini)**(0.5))+mini)**(n*N_s))+mini)**(-(2-1/(n*N_s+mini))))))

    cc = 0.5*(((1+np.sign(h))*s)+
              (1-np.sign(h))*(s+((p[1]*Theta_s_s-p[2]*Theta_r_s)*(p[3]*Alpha_s)*(p[4]*N_s)*(1-1/(p[4]*N_s+mini)))*((-1*(p[3]*Alpha_s)*-1*((h)**2)**(0.5))**((p[4]*N_s)-1))*
                              (((1+((-1*(p[3]*Alpha_s)*-1*((h)**2+mini)**(0.5))+mini)**(p[4]*N_s))+mini)**(-(2-1/(p[4]*N_s+mini))))))

    # cc = s+((p[1]*Theta_s_s-p[2]*Theta_r_s)*(p[3]*Alpha_s)*(p[4]*N_s)*(1-1/(p[4]*N_s+mini)))*((-1*(p[3]*Alpha_s)*h)**((p[4]*N_s)-1))*(((1+((-1*(p[3]*Alpha_s)*h)+mini)**(p[4]*N_s))+mini)**(-(2-1/(p[4]*N_s+mini))))
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
    # ks = p[0]
    # theta_s = p[1]
    # theta_r = p[2]
    # alpha = p[3]
    # n = p[4]
    irr = u
    state = x
    dhdt = SX.zeros(numberOfNodes)
    # dhdt = zeros(numberOfNodes)
    dz = dzList
    for i in range(0, numberOfNodes):
        # print('time: ', timeList)
        # print('nodes: ', i)
        current_state = state[i]
        if i == 0:
            bc_zl = current_state
            bc_zu = state[i + nodesInPlane]
        elif i == nodesInZ - 1:
            bc_zl = state[i - nodesInPlane]
            KzU1 = hydraulic_conductivity(current_state, p)
            bc_zu = current_state + dz * (-1 + irr / (KzU1+mini))
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
    # print(p)
    # hp, theta_p = simulate(p)

    #     hp = ((((theta_p / 100 - p[2]) / (p[1] - p[2])) ** (1. / (-(1 - 1 / p[4]))) - 1) ** (1. / p[4])) / (
    #                 -p[3])
    #
    #     he = ((((theta_e / 100 - p[2]) / (p[1] - p[2])) ** (1. / (-(1 - 1 / p[4]))) - 1) ** (1. / p[4])) / (
    #                 -p[3])
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
            obj += (theta_avg - theta_e[-(i+1)])**2
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
plb = array([Ks*0.6, Theta_s*0.6, Theta_r*0.6, Alpha*0.6, N*0.6])
pub = array([Ks*1.4, Theta_s*1.4, Theta_r*1.4, Alpha*1.4, N*1.4])
# plb = array([Ks, Theta_s*0.96, Theta_r, Alpha, N])
# pub = array([Ks*1.04, Theta_s, Theta_r*1.02, Alpha*1.05, N*1.05])

# Time interval
# # General
# ratio_t = 1
# dt = 60.0  # second
# # timeSpan = 1444  # min # 19 hour
# timeSpan = 928  # minutes
# interval = int(timeSpan*60/dt)

# Irrigation scheduled
ratio_t = 1
ratio_t_farm = 1
dt = 60/ratio_t/ratio_t_farm  # second
# timeSpan = 2683
timeSpan = 1444
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
theta_e = numpy.loadtxt('sim_results_1D_farm_robust', unpack=True)
theta_e = theta_e.transpose()
temp1 = (((theta_e / 100 - Theta_r_e) / ((Theta_s_e - Theta_r_e) + mini)) + mini)
temp2 = (temp1 ** (1. / (-(1. - (1. / (N_e + mini))) + mini)))
temp3 = ((temp2 - 1) + mini)
temp4 = (temp3 ** (1. / (N_e + mini)))
he = temp4 / ((-Alpha_e) + mini)
rand_e = zeros((timeSpan, 4))
for i in range(theta_e.shape[0]):
    for j in range(theta_e.shape[1]):
        rand_number = np.random.normal(0, 0.1)
        rand_e[i, j] = rand_number
        theta_e[i, j] = theta_e[i, j] + rand_number

# Irrigation amount
irr_amount = zeros(interval)
for i in range(0, interval):
    # if i in range(1151*ratio_t, 1217*ratio_t):
    #     irr_amount[i] = (0.004 / (pi * 0.22 * 0.22) / ((1216-1151) * 60))
    # else:
    #     irr_amount[i] = 0

    # if i in range(0, 22*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (22 * 60))
    # elif i in range(59*ratio_t, 87*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(161*ratio_t, 189*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    # elif i in range(248*ratio_t, 276*ratio_t):
    #     irr_amount[i] = (0.001 / (pi * 0.22 * 0.22) / (27 * 60))
    if i in range(int(335*ratio_t), int(361*ratio_t)):
        irr_amount[i] = (0.0001 / (pi * 0.22 * 0.22) / (25 * 60))
    else:
        irr_amount[i] = 0

# Create symbolic variables
x = SX.sym('x', numberOfNodes, 1)
u = SX.sym('u')
k = SX.sym('k', len(p0), 1)
# xe = SX.sym('xe', 4, 1)  # 4 sensors

# Create ODE and Objective functions
print("I am ready to create ODE")
f_x = RichardsEQN_1D(x, u, k)
# f_q = objective(x, k, xe)

# Create integrator
print("I am ready to create integrator")
ode = {'x': x, 'p': vertcat(u, k), 'ode': f_x}
opts = {'tf': 60/ratio_t/ratio_t_farm}  # seconds
# opts = {'tf': 60}
I = integrator('I', 'cvodes', ode, opts)  # Build casadi integrator

# All parameter sets and irrigation amount
U = irr_amount
K = MX.sym('K', 5)
XE = theta_e
GL = []
# Construct graph of integrator calls
print("I am ready to create construct graph")
J = 0  # Initial cost function
for i in range(interval):
    if i == 0:
        G, hIni = ini_state(K)  # Initial state
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
np.savetxt('sim_results_1D_farm_casadi', thetaL_numpy_array_avg)
'''''

print("I am doing creating MX")
# Allocate an NLP solver
nlp = {'x': K, 'f': J, 'g': vertcat(*GL)}  # x: Solve for P (parameters), which gives the lowest J (cost fcn), with the constraints G (propagated model)
# nlp = {'x': K, 'f': J}
# opts = {"ipopt.linear_solver":"ma97"}
# opts = {"ipopt.linear_solver": "ma57"}
opts = {"ipopt.linear_solver": "mumps"}
opts["ipopt.print_level"] = 5
opts["regularity_check"] = True
opts["verbose"] = True
opts["ipopt.tol"]=1e-05
opts['ipopt.max_iter'] = 30
print("I am ready to build")
solver = nlpsol('solver', 'ipopt', nlp, opts)

print("I am ready to solve")

sol = solver(# constraints are missing here
        lbx=plb,
        ubx=pub,
        x0=p0  # Initial guess of decision variable
      )
print (sol)
pe = sol['x'].full().squeeze()
pe[0] = pe[0]*Ks_s
pe[1] = pe[1]*Theta_s_s
pe[2] = pe[2]*Theta_r_s
pe[3] = pe[3]*Alpha_s
pe[4] = pe[4]*N_s
print ("")
print ("Estimated parameter(s) is(are): " + str(pe))
p0[0] = Ks_s
p0[1] = Theta_s_s
p0[2] = Theta_r_s
p0[3] = Alpha_s
p0[4] = N_s
print ("")
print ("Actual value(s) is(are): " + str(p0))
#
# timeList = []
# for i in range(interval):
#     current_t = i*dt
#     timeList.append(current_t)
# timeList = array(timeList, dtype='O')
#
#
#

#
# obj_fun = int(input('Enter 1 for Theta-base objective function, or 2 for h-base: '))
#
# p0 = array([Ks, Theta_s, Theta_r, Alpha, N])
# print('Initial SSE Objective: ' + str(objective(p0)))
# bnds = ((1e-10, 1e-2), (0.302, 1.0), (0.0, 0.084), (-inf, inf), (0.8, inf))
# solution = optimize.minimize(objective, p0, bounds=bnds)
# p = solution.x
# print('Final SSE Objective: ' + str(objective(p)))
# Ks = p[0]
# Theta_s = p[1]
# Theta_r = p[2]
# Alpha = p[3]
# N = p[4]
# print('Ks: ' + str(Ks))
# print('Theta_s: ' + str(Theta_s))
# print('Theta_r: ' + str(Theta_r))
# print('Alpha: ' + str(Alpha))
# print('N: ' + str(N))
#
# hi, theta_i = simulate(p0)
# hp, theta_p = simulate(p)
#
# hi = ((((theta_i / 100 - p0[2]) / (p0[1] - p0[2])) ** (1. / (-(1 - 1 / p0[4]))) - 1) ** (1. / p0[4])) / (
#                 -p0[3])
# hp = ((((theta_p / 100 - p[2]) / (p[1] - p[2])) ** (1. / (-(1 - 1 / p[4]))) - 1) ** (1. / p[4])) / (
#                 -p[3])
# hp[0] = hIni[0]
#
# plt.figure(1)
# # plt.subplot(4, 1, 1)
# plt.plot(timeList/60.0, hi[:, 0], 'y--', label=r'$h_1$ initial')
# plt.plot(timeList/60.0, he[:, 0], 'b:', label=r'$h_1$ measured')
# plt.plot(timeList/60.0, hp[:, 0], 'r-', label=r'$h_1$ optimized')
# plt.xlabel('Time (min)')
# plt.ylabel('Pressure head (m)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(2)
# # plt.subplot(4, 1, 2)
# plt.plot(timeList/60.0, hi[:, 1], 'y--', label=r'$h_2$ initial')
# plt.plot(timeList/60.0, he[:, 1], 'b:', label=r'$h_2$ measured')
# plt.plot(timeList/60.0, hp[:, 1], 'r-', label=r'$h_2$ optimized')
# plt.xlabel('Time (min)')
# plt.ylabel('Pressure head (m)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(3)
# # plt.subplot(4, 1, 3)
# plt.plot(timeList/60.0, hi[:, 2], 'y--', label=r'$h_3$ initial')
# plt.plot(timeList/60.0, he[:, 2], 'b:', label=r'$h_3$ measured')
# plt.plot(timeList/60.0, hp[:, 2], 'r-', label=r'$h_3$ optimized')
# plt.xlabel('Time (min)')
# plt.ylabel('Pressure head (m)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(4)
# # plt.subplot(4, 1, 4)
# plt.plot(timeList/60.0, hi[:, 3], 'y--', label=r'$h_4$ initial')
# plt.plot(timeList/60.0, he[:, 3], 'b:', label=r'$h_4$ measured')
# plt.plot(timeList/60.0, hp[:, 3], 'r-', label=r'$h_4$ optimized')
# plt.xlabel('Time (min)')
# plt.ylabel('Pressure head (m)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(5)
# # plt.subplot(4, 1, 1)
# plt.plot(timeList/60.0, theta_i[:, 0], 'y--', label=r'$theta_1$ initial')
# plt.plot(timeList/60.0, theta_e[:, 0], 'b:', label=r'$theta_1$ measured')
# plt.plot(timeList/60.0, theta_p[:, 0], 'r-', label=r'$theta_1$ optimized')
# plt.xlabel('Time (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(6)
# # plt.subplot(4, 1, 2)
# plt.plot(timeList/60.0, theta_i[:, 1], 'y--', label=r'$theta_2$ initial')
# plt.plot(timeList/60.0, theta_e[:, 1], 'b-', label=r'$theta_2$ measured')
# plt.plot(timeList/60.0, theta_p[:, 1], 'r-', label=r'$theta_2$ optimized')
# plt.xlabel('Time (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(7)
# # plt.subplot(4, 1, 3)
# plt.plot(timeList/60.0, theta_i[:, 2], 'y--', label=r'$theta_3$ initial')
# plt.plot(timeList/60.0, theta_e[:, 2], 'b:', label=r'$theta_3$ measured')
# plt.plot(timeList/60.0, theta_p[:, 2], 'r-', label=r'$theta_3$ optimized')
# plt.xlabel('Time (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure(8)
# # plt.subplot(4, 1, 4)
# plt.plot(timeList/60.0, theta_i[:, 3], 'y--', label=r'$theta_4$ initial')
# plt.plot(timeList/60.0, theta_e[:, 3], 'b:', label=r'$theta_4$ measured')
# plt.plot(timeList/60.0, theta_p[:, 3], 'r-', label=r'$theta_4$ optimized')
# plt.xlabel('Time (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))
np.savetxt('sim_results_1D_farm_white_noise_widerrange', rand_e)
# # io.savemat('sim_results_1D_odeint.mat', dict(y_1D_odeint=theta_i))
