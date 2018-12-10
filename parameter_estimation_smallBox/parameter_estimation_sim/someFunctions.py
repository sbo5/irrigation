def Loam():
    pars = {}
    pars['thetaR'] = 0.078
    pars['thetaS'] = 0.43
    pars['alpha'] = 0.036*100
    pars['n'] = 1.56
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 1.04/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def LoamIni():
    pars = {}
    pars['thetaR'] = 0.078*0.9
    pars['thetaS'] = 0.43* 0.9
    pars['alpha'] = 0.036*100*0.9
    pars['n'] = 1.56* 0.9
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 1.04/100/3600*0.9
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def LoamOpt():
    pars = {}
    pars['thetaR'] = 6.18438096e-02
    pars['thetaS'] = 4.25461746e-01
    pars['alpha'] = 3.71675224e+00
    pars['n'] = 1.50828922e+00
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 2.99737827e-06
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def LoamySand():
    pars = {}
    pars['thetaR'] = 0.057
    pars['thetaS'] = 0.41
    pars['alpha'] = 0.124*100
    pars['n'] = 2.28
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 14.59/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def SandyLoam():
    pars = {}
    pars['thetaR'] = 0.065
    pars['thetaS'] = 0.41
    pars['alpha'] = 0.075*100
    pars['n'] = 1.89
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 4.42/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def SiltLoam():
    pars = {}
    pars['thetaR'] = 0.067
    pars['thetaS'] = 0.45
    pars['alpha'] = 0.020*100
    pars['n'] = 1.41
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 0.45/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


# ----------------------------------------------------------------------------------------------------------------------
# Define the geometry
# ----------------------------------------------------------------------------------------------------------------------
ratio_x = 1
ratio_y = 1
ratio_z = 1
ratio_ft_to_m = 0.3048  # m/ft
lengthOfXinFt = 2  # feet
lengthOfYinFt = 4  # feet
lengthOfZinFt = 0.7874  # feet [In order to use 2 sensors of the probe, the minimum soil depth is 150+150+41=341mm or 1.11877ft)]
                         # The maximum height is 60.50 cm or 1.98 ft
lengthOfX = lengthOfXinFt * ratio_ft_to_m  # meter. Should be around 61 cm
lengthOfY = lengthOfYinFt * ratio_ft_to_m  # meter. Should be around 121.8 cm
# lengthOfZ = lengthOfZinFt * ratio_ft_to_m  # meter. Should be around 60.5 cm
lengthOfZ = 0.24  # height of soil in the small box

nodesInX = int(2*ratio_x)
nodesInY = int(2*ratio_y)
nodesInZ = int(24*ratio_z)  # Define the nodes

intervalInX = nodesInX
intervalInY = nodesInY
intervalInZ = nodesInZ  # The distance between the boundary to its adjacent states is 1/2 of delta_z

nodesInPlane = nodesInX * nodesInY
numberOfNodes = nodesInZ*nodesInPlane

dx = lengthOfX/intervalInX
dy = lengthOfY/intervalInY
dz = lengthOfZ/intervalInZ

# ----------------------------------------------------------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------------------------------------------------------
numberOfSensors = 1
layersOfSensors = np.array([0,5,20,nodesInZ])  # beginning = top & end = bottom
# C matrix
start = layersOfSensors[1] * ratio_z
end = layersOfSensors[2] * ratio_z
difference = end - start
CMatrix = np.zeros((numberOfSensors, numberOfNodes))
for i in range(0, numberOfSensors):
    if i == 0:
        CMatrix[i][(start - 1 * ratio_z) * nodesInPlane: end * nodesInPlane] = 1. / (
                    (end - (start - 1 * ratio_z)) * nodesInPlane)
    else:
        CMatrix[i][start * nodesInPlane: end * nodesInPlane] = 1. / ((end - start) * nodesInPlane)
    start += difference * ratio_z
    end += difference * ratio_z
# ----------------------------------------------------------------------------------------------------------------------
# Time interval
# ----------------------------------------------------------------------------------------------------------------------
ratio_t = 1
dt = 60.0*ratio_t  # second
timeSpan = 15
interval = int(timeSpan*60/dt)

timeList_original = np.arange(0, timeSpan+1)*dt/ratio_t

timeList = np.arange(0, interval+1)*dt

# ----------------------------------------------------------------------------------------------------------------------
# Inputs: irrigation
# ----------------------------------------------------------------------------------------------------------------------
irrigation = np.zeros((len(timeList), nodesInX, nodesInY))
for i in range(0, len(irrigation)):  # 1st node is constant for 1st temporal element. The last node is the end of the last temporal element.
    if i in range(0, 180):
        irrigation[i] = -0.050/86400
    elif i in range(180, 540):
        irrigation[i] = -0.010/86400
    else:
        irrigation[i] = 0


def hFun(theta, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - pars['thetaR']) / (pars['thetaS'] - pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-pars['m'] + pars['mini']))
              - 1) + pars['mini']) ** (1. / (pars['n'] + pars['mini']))) / (-pars['alpha'] + pars['mini'])
    return psi
