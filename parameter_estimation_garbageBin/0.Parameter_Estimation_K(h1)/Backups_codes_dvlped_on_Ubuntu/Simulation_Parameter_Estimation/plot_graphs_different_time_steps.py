from __future__ import (absolute_import, print_function, division, unicode_literals)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
# from PyFoam.Error import error
# from PyFoam.Execution.BasicRunner import BasicRunner
# from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import mpctools as mpc
from casadi import *
from math import *
import matplotlib.pyplot as plt


# Parameters
# Experimental parameters
Ks_e = 0.00000288889  # [m/s]
Theta_s_e = 0.43
Theta_r_e = 0.078
Alpha_e = 3.6
N_e = 1.56
mini = 1e-100
# Irrigation scheduled
ratio_t = 2
dt = 60.0/ratio_t  # second
timeSpan = 1444  # min # 19 hour
interval = int(timeSpan*60/dt)

timeList = []
for i in range(interval):
    current_t = i*dt
    timeList.append(current_t)
timeList = array(timeList, dtype='O')

timeList_5776 = []
for i in range(5776):
    current_t = i*15
    timeList_5776.append(current_t)
timeList_5776 = array(timeList_5776, dtype='O')

timeList_original = []
for i in range(timeSpan):
    current_t = i*dt*ratio_t
    timeList_original.append(current_t)
timeList_original = array(timeList_original, dtype='O')

# Reading data from the file
theta_e_1444 = numpy.loadtxt('sim_results_1D_farm_robust', unpack=True)
theta_e_1444 = theta_e_1444.transpose()
temp1 = (((theta_e_1444 / 100 - Theta_r_e) / ((Theta_s_e - Theta_r_e) + mini)) + mini)
temp2 = (temp1 ** (1. / (-(1. - (1. / (N_e + mini))) + mini)))
temp3 = ((temp2 - 1) + mini)
temp4 = (temp3 ** (1. / (N_e + mini)))
he_1444 = temp4 / ((-Alpha_e) + mini)

theta_e_1444_casadi = numpy.loadtxt('sim_results_1D_farm_casadi', unpack=True)
theta_e_1444_casadi = theta_e_1444_casadi.transpose()
temp1 = (((theta_e_1444_casadi / 100 - Theta_r_e) / ((Theta_s_e - Theta_r_e) + mini)) + mini)
temp2 = (temp1 ** (1. / (-(1. - (1. / (N_e + mini))) + mini)))
temp3 = ((temp2 - 1) + mini)
temp4 = (temp3 ** (1. / (N_e + mini)))
he_1444_casadi = temp4 / ((-Alpha_e) + mini)

# Reading data from the file
theta_e_2888 = numpy.loadtxt('sim_results_1D_farm_robust_2888', unpack=True)
theta_e_2888 = theta_e_2888.transpose()
temp1 = (((theta_e_2888 / 100 - Theta_r_e) / ((Theta_s_e - Theta_r_e) + mini)) + mini)
temp2 = (temp1 ** (1. / (-(1. - (1. / (N_e + mini))) + mini)))
temp3 = ((temp2 - 1) + mini)
temp4 = (temp3 ** (1. / (N_e + mini)))
he_2888 = temp4 / ((-Alpha_e) + mini)

# Reading data from the file
theta_e_2888_casadi = numpy.loadtxt('sim_results_1D_farm_casadi_2888', unpack=True)
theta_e_2888_casadi = theta_e_2888_casadi.transpose()
temp1 = (((theta_e_2888_casadi / 100 - Theta_r_e) / ((Theta_s_e - Theta_r_e) + mini)) + mini)
temp2 = (temp1 ** (1. / (-(1. - (1. / (N_e + mini))) + mini)))
temp3 = ((temp2 - 1) + mini)
temp4 = (temp3 ** (1. / (N_e + mini)))
he_2888_casadi = temp4 / ((-Alpha_e) + mini)

# Reading data from the file
theta_e_5776 = numpy.loadtxt('sim_results_1D_farm_robust_5776', unpack=True)
theta_e_5776 = theta_e_5776.transpose()
temp1 = (((theta_e_5776 / 100 - Theta_r_e) / ((Theta_s_e - Theta_r_e) + mini)) + mini)
temp2 = (temp1 ** (1. / (-(1. - (1. / (N_e + mini))) + mini)))
temp3 = ((temp2 - 1) + mini)
temp4 = (temp3 ** (1. / (N_e + mini)))
he_5776 = temp4 / ((-Alpha_e) + mini)

# Reading data from the file
theta_e_5776_casadi = numpy.loadtxt('sim_results_1D_farm_casadi_5776', unpack=True)
theta_e_5776_casadi = theta_e_5776_casadi.transpose()
temp1 = (((theta_e_5776_casadi / 100 - Theta_r_e) / ((Theta_s_e - Theta_r_e) + mini)) + mini)
temp2 = (temp1 ** (1. / (-(1. - (1. / (N_e + mini))) + mini)))
temp3 = ((temp2 - 1) + mini)
temp4 = (temp3 ** (1. / (N_e + mini)))
he_5776_casadi = temp4 / ((-Alpha_e) + mini)

plt.figure(9)
# plt.subplot(4, 1, 1)
# plt.plot(timeList_original/60.0, theta_e_1444[:, 0], 'y-', label=r'$theta_1$ measured 1444')
# plt.plot(timeList_original/60.0, theta_e_1444_casadi[:, 0], 'r--', label=r'$theta_1$ measured 1444 casadi')
# plt.plot(timeList/60.0, theta_e_2888[:, 0], 'r--', label=r'$theta_1$ optimized 2888')
# plt.plot(timeList/60.0, theta_e_2888_casadi[:, 0], 'b:', label=r'$theta_1$ optimized 2888 casadi')
plt.plot(timeList_5776/60.0, theta_e_5776[:, 0], 'b:', label=r'$theta_1$ optimized 5776')
plt.plot(timeList_5776/60.0, theta_e_5776_casadi[:, 0], 'y-', label=r'$theta_1$ optimized 5776 casadi')
plt.xlabel('Time (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

plt.figure(10)
# plt.subplot(4, 1, 1)
# plt.plot(timeList_original/60.0, he_1444[:, 0], 'y-', label=r'$h_1$ measured 1444')
# plt.plot(timeList_original/60.0, he_1444_casadi[:, 0], 'r--', label=r'$h_1$ measured 1444 casadi')
# plt.plot(timeList/60.0, he_2888[:, 0], 'r--', label=r'$h_1$ optimized 2888')
# plt.plot(timeList/60.0, he_2888_casadi[:, 0], 'b:', label=r'$h_1$ optimized 2888 casadi')
plt.plot(timeList_5776/60.0, he_5776[:, 0], 'b:', label=r'$h_1$ optimized 5776')
plt.plot(timeList_5776/60.0, he_5776_casadi[:, 0], 'y-', label=r'$h_1$ optimized 5776 casadi')
plt.xlabel('Time (min)')
plt.ylabel('Head pressure (m)')
plt.legend(loc='best')
plt.show()