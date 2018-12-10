"""
Created on Thu Aug 27 09:40:31 2018

@author: Song Bo (sbo@ualberta.ca)

This is a Richards equation parameter estimation example using orthogonal collocation on finite element.
dxdt = []

 = g(x,z,u,p)

The parameter to be estimated is

"""

# from __future__ import (print_function)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
from casadi import *
import matplotlib.pyplot as plt

print("I am starting up")
start_time = time.time()

# tf = 60.*5695
# nk = 5695
tf = 60.*4360
nk = 4360
ratio_t = 1
h = (tf/nk)*ratio_t

timeList_original = np.arange(0, nk+1)*h/ratio_t

h_exp = np.zeros((len(timeList_original), 4))  # four sensors
theta_exp = np.zeros((len(timeList_original), 4))
# with open('Data/exp_data_noirr_4041.dat', 'r') as f:
# with open('Data/exp_data_4day_5696.dat', 'r') as f:
# with open('Data/exp_data_5L_1444.dat', 'r') as f:
# with open('Data/exp_data_4L_2683.dat', 'r') as f:
with open('Data/exp_data_5L_4L_4361.dat', 'r') as f:
    wholeFile = f.readlines()
    for index, line in enumerate(wholeFile):
        oneLine = line.rstrip().split(",")
        oneLine = oneLine[1:5]
        theta_temp = []
        for index1, item in enumerate(oneLine):
            item = float(item)
            theta_temp.append(item)
        theta_temp = array(theta_temp, dtype='O')
        theta_exp[index] = theta_temp
y_exp = theta_exp      # Experimental measurments states/outputs

plt.figure()
plt.plot(timeList_original/h*ratio_t, y_exp[:, 0], ':', label=r'$\theta_{1,exp}$')
plt.plot(timeList_original/h*ratio_t, y_exp[:, 1], ':', label=r'$\theta_{2,exp}$')
plt.plot(timeList_original/h*ratio_t, y_exp[:, 2], ':', label=r'$\theta_{3,exp}$')
plt.plot(timeList_original/h*ratio_t, y_exp[:, 3], ':', label=r'$\theta_{4,exp}$')

# plt.plot(timeList/h*ratio_t, theta_i[:, 0], 'y--', label=r'$theta_1$ initial')
# plt.plot(timeList/h*ratio_t, theta_opt[:, 0], 'r--', label=r'$theta_1$ optimized')
plt.xlabel('Time, t (min)')
plt.ylabel('Water content (%)')
plt.legend(loc='best')
plt.show()

# plt.figure()
# plt.plot(timeList_original/h*ratio_t, y_exp[:, 0], 'b-.', label=r'$theta_1$ experimental')
# # plt.plot(timeList/h*ratio_t, theta_i[:, 0], 'y--', label=r'$theta_1$ initial')
# # plt.plot(timeList/h*ratio_t, theta_opt[:, 0], 'r--', label=r'$theta_1$ optimized')
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# plt.plot(timeList_original/h*ratio_t, y_exp[:, 1], 'b-.', label=r'$theta_2$ experimental')
# # plt.plot(timeList/h*ratio_t, theta_i[:, 1], 'y--', label=r'$theta_2$ initial')
# # plt.plot(timeList/h*ratio_t, theta_opt[:, 1], 'r--', label=r'$theta_2$ optimized')
#
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# plt.plot(timeList_original/h*ratio_t, y_exp[:, 2], 'b-.', label=r'$theta_3$ experimental')
# # plt.plot(timeList/h*ratio_t, theta_i[:, 2], 'y--', label=r'$theta_3$ initial')
# # plt.plot(timeList/h*ratio_t, theta_opt[:, 2], 'r--', label=r'$theta_3$ optimized')
#
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()
#
# plt.figure()
# plt.plot(timeList_original/h*ratio_t, y_exp[:, 3], 'b-.', label=r'$theta_4$ experimental')
# # plt.plot(timeList/h*ratio_t, theta_i[:, 3], 'y--', label=r'$theta_4$ initial')
# # plt.plot(timeList/h*ratio_t, theta_opt[:, 3], 'r--', label=r'$theta_4$ optimized')
#
# plt.xlabel('Time, t (min)')
# plt.ylabel('Water content (%)')
# plt.legend(loc='best')
# plt.show()

print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))