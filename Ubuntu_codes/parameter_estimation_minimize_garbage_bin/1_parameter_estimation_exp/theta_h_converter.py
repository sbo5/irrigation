from __future__ import (absolute_import, print_function, division, unicode_literals)
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array


# Parameters
# Initial parameters
Ks = 0.00000288889  # [m/s]
Theta_s = 0.43
Theta_r = 0.078
Alpha = 3.6
N = 1.56

thetaList = []
c = False
c0 = 1
with open('theta', 'r') as f:
    whole = f.readlines()
    for index, line in enumerate(whole):
        if line == '(\n':
            c = True
            c0 = 0
        if line in [')\n', ');', ');\n']:
            c = False
            break
        if c == True & c0 == 1:
            line.rstrip()
            line_float = float(line)
            thetaList.append(line_float)
        c0 = 1
    thetaList = array(thetaList, dtype='O')
    thetaList = thetaList.ravel()

temp1 = ((thetaList - Theta_r) / (Theta_s - Theta_r))
temp2 = (temp1 ** (1. / (-(1. - (1. / N)))))
temp3 = (temp2 - 1)
temp4 = (temp3 ** (1. / N))
hList = temp4 / (-Alpha)



