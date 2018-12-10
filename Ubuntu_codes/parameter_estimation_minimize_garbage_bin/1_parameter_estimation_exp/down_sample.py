from __future__ import (absolute_import, print_function, division, unicode_literals)
from numpy import diag, zeros, ones, dot, copy, mean, asarray, array
import matplotlib.pyplot as plt

# Parameters
# Initial parameters
Ks = 0.00000288889  # [m/s]
Theta_s = 0.43
Theta_r = 0.078
Alpha = 3.6
N = 1.56

# Time interval
dt = 60.0  # second
timeSpan = 1444  # min # 19 hour
interval = int(timeSpan*60/dt)

timeList = []
for i in range(interval):
    current_t = i*dt
    timeList.append(current_t)
timeList = array(timeList, dtype='O')

he = zeros((len(timeList), 4))  # four sensors
theta_e = zeros((len(timeList), 4))
with open('exp_data_original.dat', 'r') as f:
    whole = f.readlines()
    for index, line in enumerate(whole):
        one_line = line.rstrip().split(",")
        one_line = one_line[1:5]
        h_temp = []
        theta_temp = []
        for index1, item in enumerate(one_line):
            item = float(item)
            theta_temp.append(item)
            temp1 = ((item/100 - Theta_r) / (Theta_s - Theta_r))
            temp2 = (temp1 ** (1. / (-(1. - (1. / N)))))
            temp3 = (temp2 - 1)
            temp4 = (temp3 ** (1. / N))
            item = temp4 / (-Alpha)
            h_temp.append(item)
        h_temp = array(h_temp, dtype='O')
        theta_temp = array(theta_temp, dtype='O')
        he[index] = h_temp
        theta_e[index] = theta_temp

numberOfSamples = 144
sampleInterval = int(1440/numberOfSamples)
he_downsample = zeros((numberOfSamples, 4))
theta_e_downsample = zeros((numberOfSamples, 4))
timeList_downsample = zeros((numberOfSamples, 1))

for i in range(numberOfSamples):
    he_downsample[i] = he[i*sampleInterval]
    theta_e_downsample[i] = theta_e[i*sampleInterval]
    timeList_downsample[i] = timeList[i*sampleInterval]

plt.figure(1)
# plt.subplot(4, 1, 1)
plt.plot(timeList/60.0, he[:, 0], 'y:', label=r'$h_1$ original')
plt.plot(timeList_downsample/60.0, he_downsample[:, 0], 'b--', label=r'$h_1$ down sampled')
plt.xlabel('Time (min)')
plt.ylabel('Pressure head (m)')
plt.legend(loc='best')
plt.show()

plt.figure(2)
# plt.subplot(4, 1, 2)
plt.plot(timeList/60.0, theta_e[:, 0], 'y:', label=r'$theta_1$ original')
plt.plot(timeList_downsample/60.0, theta_e_downsample[:, 0], 'b--', label=r'$theta_1$ down sampled')
plt.xlabel('Time (min)')
plt.ylabel('Pressure head (m)')
plt.legend(loc='best')
plt.show()