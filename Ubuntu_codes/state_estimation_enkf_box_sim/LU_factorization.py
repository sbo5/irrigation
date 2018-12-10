#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:04:21 2018

@author: sbo
"""
import numpy as np
from scipy.linalg import lu
from numpy import dot, zeros, ones, eye, outer
from numpy.linalg import solve, matrix_rank


P_zz=np.array([[2, 0, 5, 3], [1, 2, 2, 1],[3, 5, 2, 1],[4, 3, 5, 1]])
sln = np.array([[1,2,1,2],[3,4,3,4],[1,2,2,1],[1,2,1,1]])

P_xz = dot(P_zz, sln)

solution = solve(P_zz, P_xz)

dim_x = len(P_xz)
y = np.zeros(P_xz.shape)
Pe, L, U = lu(P_zz)
P_xzP = dot(Pe, P_xz)
y[0] = P_xzP[0]
for i in range(1, dim_x):
    alpha = L[i][0:i]
    beta = P_xzP[i]
    gamma = y[0:i]
    temp = dot(alpha, gamma)
    y[i] = beta - temp

K = np.zeros(P_xz.shape)
Ulast = U[dim_x-1][-1:]
K[dim_x-1] = y[dim_x-1]/Ulast
for i in range(dim_x-2, -1, -1):
    alpha1 = U[i][i+1:]
    beta1 = y[i]
    gamma1 = K[i+1:]
    temp1 = dot(alpha1, gamma1)
    temp2 = beta1 - temp1
    Udiag = U[i][i]
    K[i] = temp2/Udiag
