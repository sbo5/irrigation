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

start_time = time.time()
# Loam
Ks_loam = 1.04/100/3600  # [m/s]
Theta_s_loam = 0.43
Theta_r_loam = 0.078
Alpha_loam = 0.036*100
N_loam = 1.56

# Loamy sand
Ks_loamysand = 14.59/100/3600  # [m/s]
Theta_s_loamysand = 0.41
Theta_r_loamysand = 0.057
Alpha_loamysand = 0.124*100
N_loamysand = 2.28

Ks_loamysand1 = 3/100/3600  # [m/s]
Theta_s_loamysand1 = 0.39
Theta_r_loamysand1 = 0.08
Alpha_loamysand1 = 0.1*100
N_loamysand1 = 1.56

Ks_loamysand2 = 3/100/3600  # [m/s]
Theta_s_loamysand2 = 0.39
Theta_r_loamysand2 = 0.08
Alpha_loamysand2 = 0.1*100*1.1
N_loamysand2 = 1.56*1.1

Ks_loamysand3 = 3/100/3600  # [m/s]
Theta_s_loamysand3 = 0.39
Theta_r_loamysand3 = 0.08
Alpha_loamysand3 = 0.1*100*1.15
N_loamysand3 = 1.56*1.15

Ks_loamysand4 = 3/100/3600  # [m/s]
Theta_s_loamysand4 = 0.39
Theta_r_loamysand4 = 0.08
Alpha_loamysand4 = 0.1*100*1.2
N_loamysand4 = 1.56*1.2


p_loam = array([Ks_loam, Theta_s_loam, Theta_r_loam, Alpha_loam, N_loam])
p_loamysand = array([Ks_loamysand, Theta_s_loamysand, Theta_r_loamysand, Alpha_loamysand, N_loamysand])
p_loamysand1 = array([Ks_loamysand1, Theta_s_loamysand1, Theta_r_loamysand1, Alpha_loamysand1, N_loamysand1])
p_loamysand2 = array([Ks_loamysand2, Theta_s_loamysand2, Theta_r_loamysand2, Alpha_loamysand2, N_loamysand2])
p_loamysand3 = array([Ks_loamysand3, Theta_s_loamysand3, Theta_r_loamysand3, Alpha_loamysand3, N_loamysand3])
p_loamysand4 = array([Ks_loamysand4, Theta_s_loamysand4, Theta_r_loamysand4, Alpha_loamysand4, N_loamysand4])

S = 0.0001  # [per m]
PET = 0.0000000070042  # [per sec]
mini = 1e-100

h = np.arange(-10, 1)
# hini = array([-0.0776214,  -0.53626989, -0.55036712, -0.41337759])
# theta_ini = 100 * ((p_loamysand[1] - p_loamysand[2]) * ((1 + (-p_loamysand[3] * hini + mini) ** p_loamysand[4]) + mini) ** (-(1 - 1 / (p_loamysand[4] + mini))) + p_loamysand[2])

# theta_loam = 100 * ((p_loam[1] - p_loam[2]) * ((1 + (-p_loam[3] * h) ** p_loam[4])) ** (-(1 - 1 / (p_loam[4]))) + p_loam[2])
theta_loamysand = 100 * ((p_loamysand[1] - p_loamysand[2]) * ((1 + (-p_loamysand[3] * h + mini) ** p_loamysand[4]) + mini) ** (-(1 - 1 / (p_loamysand[4] + mini))) + p_loamysand[2])
theta_loamysand1 = 100 * ((p_loamysand1[1] - p_loamysand1[2]) * ((1 + (-p_loamysand1[3] * h + mini) ** p_loamysand1[4]) + mini) ** (-(1 - 1 / (p_loamysand1[4] + mini))) + p_loamysand1[2])
theta_loamysand2 = 100 * ((p_loamysand2[1] - p_loamysand2[2]) * ((1 + (-p_loamysand2[3] * h + mini) ** p_loamysand2[4]) + mini) ** (-(1 - 1 / (p_loamysand2[4] + mini))) + p_loamysand2[2])
theta_loamysand3 = 100 * ((p_loamysand3[1] - p_loamysand3[2]) * ((1 + (-p_loamysand3[3] * h + mini) ** p_loamysand3[4]) + mini) ** (-(1 - 1 / (p_loamysand3[4] + mini))) + p_loamysand3[2])
theta_loamysand4 = 100 * ((p_loamysand4[1] - p_loamysand4[2]) * ((1 + (-p_loamysand4[3] * h + mini) ** p_loamysand4[4]) + mini) ** (-(1 - 1 / (p_loamysand4[4] + mini))) + p_loamysand4[2])

plt.figure(1)
# plt.plot(h, theta_loam,'r--', label='loam')
plt.plot(h, theta_loamysand,'y-', label=r'loamy sand, $\alpha$ = '+str(Alpha_loamysand)+', n = '+str(N_loamysand))
# plt.plot(h, theta_loamysand1, 'b-', label=r'loamy sand1, $\alpha$ = '+str(Alpha_loamysand1)+', n = '+str(N_loamysand1))
# plt.plot(h, theta_loamysand2, 'b--', label=r'loamy sand2, $\alpha$ = '+str(Alpha_loamysand2)+', n = '+str(N_loamysand2))
# plt.plot(h, theta_loamysand3, 'b:', label=r'loamy sand3, $\alpha$ = '+str(Alpha_loamysand3)+', n = '+str(N_loamysand3))
# plt.plot(h, theta_loamysand4, 'g:', label=r'loamy sand4, $\alpha$ = '+str(Alpha_loamysand4)+', n = '+str(N_loamysand4))

plt.xlabel('Pressure head (m)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')

# ax = plt.gca()
# ax.invert_yaxis()
plt.show()