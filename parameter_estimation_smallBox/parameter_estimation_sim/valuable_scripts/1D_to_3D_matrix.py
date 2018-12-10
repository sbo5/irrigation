from __future__ import (print_function, division)
from scipy import io, optimize, integrate
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array, interp
from scipy.linalg import lu
from numpy.linalg import inv, matrix_rank, cond, cholesky, norm
import time
import csv
from casadi import *
# from math import *
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# Define the geometry
# ----------------------------------------------------------------------------------------------------------------------
ratio_x = 1
ratio_y = 1
ratio_z = 1
ratio_ft_to_m = 0.3048  # m/ft
lengthOfXinFt = 2  # feet
lengthOfYinFt = 4  # feet
lengthOfZinFt = 1.9849  # feet [In order to use 2 sensors of the probe, the minimum soil depth is 150+150+41=341mm or 1.11877ft)]
                         # The maximum height is 60.50 cm or 1.98 ft
lengthOfX = lengthOfXinFt * ratio_ft_to_m  # meter. Should be around 61 cm
lengthOfY = lengthOfYinFt * ratio_ft_to_m  # meter. Should be around 121.8 cm
lengthOfZ = lengthOfZinFt * ratio_ft_to_m  # meter. Should be around 60.5 cm

nodesInX = int(12*ratio_x)
nodesInY = int(24*ratio_y)
nodesInZ = int(12*ratio_z)  # Define the nodes

intervalInX = nodesInX
intervalInY = nodesInY
intervalInZ = nodesInZ  # The distance between the boundary to its adjacent states is 1/2 of delta_z

nodesInPlane = nodesInX * nodesInY
numberOfNodes = nodesInZ*nodesInPlane


def hFun(theta, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - pars['thetaR']) / (pars['thetaS'] - pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-pars['m'] + pars['mini']))
              - 1) + pars['mini']) ** (1. / (pars['n'] + pars['mini']))) / (-pars['alpha'] + pars['mini'])
    return psi


# Calculated the initial state
def ini_state(thetaIni, p):
    psiIni = hFun(thetaIni, p)
    # Approach 1: States in the same section are the same
    layerForEachSensor = [0, 3, 6, 9, nodesInZ]  # 1 sensor covers 0to8th layers.
    hMatrix = np.zeros(shape=(intervalInX, intervalInY, intervalInZ))
    hMatrix[:,:,int(layerForEachSensor[0]*ratio_z):int(layerForEachSensor[1]*ratio_z)] = psiIni[0]  # 1st section has 8 states
    hMatrix[:,:,int(layerForEachSensor[1]*ratio_z):int(layerForEachSensor[2]*ratio_z)] = psiIni[1]  # After, each section has 7 states
    hMatrix[:,:,int(layerForEachSensor[2]*ratio_z):int(layerForEachSensor[3]*ratio_z)] = psiIni[2]
    hMatrix[:,:,int(layerForEachSensor[3]*ratio_z):int(layerForEachSensor[4]*ratio_z)] = psiIni[3]
    # # Approach 2: States in the same section are the same, overlap values are averaged
    # hMatrix = np.zeros(numberOfNodes)
    # hMatrix[int(0):int(1)] = psiIni[0]  # 1st section has 8 states
    # hMatrix[int(1):int(2)] = psiIni[1]  # After, each section has 7 states
    # hMatrix[int(2):int(3)] = psiIni[2]
    # hMatrix[int(3):numberOfNodes] = psiIni[3]
    return hMatrix, psiIni