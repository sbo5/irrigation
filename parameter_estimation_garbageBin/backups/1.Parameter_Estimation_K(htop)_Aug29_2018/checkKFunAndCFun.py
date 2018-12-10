import matplotlib.pyplot as plt
import numpy as np
from casadi import *


# ----------------------------------------------------------------------------------------------------------------------
# Refrence codes
# ----------------------------------------------------------------------------------------------------------------------
def thetaFun_ref(psi,pars):
    if psi>=0.:
        Se = 1.
    else:
        Se=(1+abs(psi*pars['alpha'])**pars['n'])**(-pars['m'])
    return (pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se)*100


def KFun_ref(psi,pars):
    if psi>=0.:
        Se=1.
    else:
        Se=(1+abs(psi*pars['alpha'])**pars['n'])**(-pars['m'])
    return pars['Ks']*Se**pars['neta']*(1-(1-Se**(1/pars['m']))**pars['m'])**2


def CFun_ref(psi,pars):
    if psi>=0.:
        Se=1.
    else:
        Se=(1+abs(psi*pars['alpha'])**pars['n'])**(-pars['m'])
    dSedh=pars['alpha']*pars['m']/(1-pars['m'])*Se**(1/pars['m'])*(1-Se**(1/pars['m']))**pars['m']
    return Se*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh

# ----------------------------------------------------------------------------------------------------------------------
# Copy from 'model_TopToBot'
# ----------------------------------------------------------------------------------------------------------------------
def hFun(theta, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - pars['thetaR']) / (pars['thetaS'] - pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-pars['m'] + pars['mini']))
              - 1) + pars['mini']) ** (1. / (pars['n'] + pars['mini']))) / (-pars['alpha'] + pars['mini'])
    return psi


def thetaFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    theta = 100*(pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se)
    theta = theta.full().ravel()
    return theta


def KFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    K = pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/(pars['m']+pars['mini'])))+pars['mini'])**pars['m']+pars['mini'])**2
    K = K.full().ravel()
    return K


def CFun(psi,pars):
    Se = if_else(psi>=0., 1., (1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    dSedh=pars['alpha']*pars['m']/(1-pars['m']+pars['mini'])*(Se+pars['mini'])**(1/(pars['m']+pars['mini']))*(1-(Se+pars['mini'])**(1/(pars['m']+pars['mini']))+pars['mini'])**pars['m']
    C = Se*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh
    C = C.full().ravel()
    return C


# ----------------------------------------------------------------------------------------------------------------------
# Copy from para_est_collocation
# ----------------------------------------------------------------------------------------------------------------------
def hFun1(theta, p, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - p[2]*pars['thetaR']) / (p[1]*pars['thetaS'] - p[2]*pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-(1-1/(p[4]*pars['n']+pars['mini'])) + pars['mini']))
              - 1) + pars['mini']) ** (1. / (p[4]*pars['n'] + pars['mini']))) / (-p[3]*pars['alpha'] + pars['mini'])
    return psi


def thetaFun1(psi, p, pars):
    Se = if_else(psi>=0., 1., (1+fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    theta = 100*(p[2]*pars['thetaR']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*Se)
    theta = theta.full().ravel()
    return theta

# This one does not use fabs -------------------------------------------------------------------------------------------
def thetaFun2(psi, p, pars):
    Se = if_else(psi>=0., 1., (1+(((psi**2)**0.5)*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    theta = 100*(p[2]*pars['thetaR']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*Se)
    theta = theta.full().ravel()
    return theta
# ----------------------------------------------------------------------------------------------------------------------


def KFun1(psi, p, pars):
    temp0 = (1+fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini'])))
    Se = if_else(psi>=0., 1., temp0)
    Se0 = Se.full().ravel()
    K = p[0]*pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])**2
    K = K.full().ravel()
    return K



def CFun1(psi, p, pars):
    Se = if_else(psi>=0., 1., (1+fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    dSedh=p[3]*pars['alpha']*(1-1/(p[4]*pars['n']+pars['mini']))/(1-(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])*(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini']))*(1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini']))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))
    C = Se*pars['Ss']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*dSedh
    C = C.full().ravel()
    return C


# ----------------------------------------------------------------------------------------------------------------------
# Tanh or tanh
# ----------------------------------------------------------------------------------------------------------------------
def hFun_tanh1(theta, p, pars):  # Assume all theta are smaller than theta_s
    psi = (((((theta/100 - p[2]*pars['thetaR']) / (p[1]*pars['thetaS'] - p[2]*pars['thetaR'] + pars['mini']) + pars['mini']) ** (1. / (-(1-1/(p[4]*pars['n']+pars['mini'])) + pars['mini']))
              - 1) + pars['mini']) ** (1. / (p[4]*pars['n'] + pars['mini']))) / (-p[3]*pars['alpha'] + pars['mini'])
    return psi


def thetaFun_tanh1(psi, p, pars):
    Se = 0.5*((1+tanh(psi))*1.+(1-tanh(psi))*((1+fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini'])))))
    theta = 100*(p[2]*pars['thetaR']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*Se)
    # theta = theta.full().ravel()
    return theta


def KFun_tanh1(psi, p, pars):
    temp1 = ((1+fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    Se = 0.5*((1+(tanh(psi+0.01)))*1.+(1-(tanh(psi+0.01)))*temp1)
    # Se1 = Se.full().ravel()
    K = p[0]*pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])**2
    # K = K.full().ravel()
    return K



def hydraulic_conductivity(h, pars):
    term3 = (1+((-1*pars['alpha']*-1*(h**2)**(1./2.))**pars['n']))
    term4 = (term3**(-(1-1/pars['n'])))
    term5 = term4**(1./2.)
    term6 = term4**(pars['n']/(pars['n']-1))
    term7 = (1-term6)**(1-1/pars['n'])
    term8 = ((1-term7)**2)
    term1 = ((1 + tanh(h+0.01)) * pars['Ks'])
    term2 = (1-tanh(h+0.01))*pars['Ks']*term5*term8
    term0 = (term1+term2)
    hc = 0.5*term0
    return hc


def KFun_he(psi,pars):
    he = 0.15  # [m]
    Sc = ((1+abs(he*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))
    Se = if_else(psi>=he, 1., ((1+abs(psi*pars['alpha']+pars['mini'])**pars['n']+pars['mini'])**(-pars['m']))/Sc)
    K = if_else(Se>=1, pars['Ks'], pars['Ks']*(Se+pars['mini'])**pars['neta']*((1-((1-(Se*Sc+pars['mini'])**(1/(pars['m']+pars['mini'])))+pars['mini'])**pars['m']+pars['mini'])/(1-((1-(Sc+pars['mini'])**(1/(pars['m']+pars['mini'])))+pars['mini'])**pars['m']+pars['mini']))**2)
    K = K.full().ravel()
    return K


def KFun_sign1(psi, p, pars):
    temp1 = ((1+fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini']))))
    Se = 0.5*((1+sign(psi))*1.+(1-sign(psi))*temp1)
    # Se1 = Se.full().ravel()
    K = p[0]*pars['Ks']*(Se+pars['mini'])**pars['neta']*(1-((1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])**2
    # K = K.full().ravel()
    return K





def CFun_tanh1(psi, p, pars):
    Se = 0.5*((1+tanh(psi))*1.+(1-tanh(psi))*((1+fabs(psi*p[3]*pars['alpha']+pars['mini'])**(p[4]*pars['n'])+pars['mini'])**(-(1-1/(p[4]*pars['n']+pars['mini'])))))
    dSedh=p[3]*pars['alpha']*(1-1/(p[4]*pars['n']+pars['mini']))/(1-(1-1/(p[4]*pars['n']+pars['mini']))+pars['mini'])*(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini']))*(1-(Se+pars['mini'])**(1/((1-1/(p[4]*pars['n']+pars['mini']))+pars['mini']))+pars['mini'])**(1-1/(p[4]*pars['n']+pars['mini']))
    C = Se*pars['Ss']+(p[1]*pars['thetaS']-p[2]*pars['thetaR'])*dSedh
    # C = C.full().ravel()
    return C


# ----------------------------------------------------------------------------------------------------------------------
# End
# ----------------------------------------------------------------------------------------------------------------------
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
    pars['mini'] = 1.0e-20
    return pars



def HygieneSandstone():
  pars={}
  pars['thetaR']=0.153
  pars['thetaS']=0.25
  pars['alpha']=0.79
  pars['n']=10.4
  pars['m']=1-1/pars['n']
  pars['Ks']=1.08
  pars['neta']=0.5
  pars['Ss']=0.000001
  pars['mini'] = 1.0e-20
  return pars


pars = Loam()
p = [1, 1, 1, 1, 1]
psi = np.linspace(-400, 10, 10000)
# psi = np.array([0.5])
theta_ref = []
C_ref = []
K_ref = []
for index, item in enumerate(psi):
    theta_r = thetaFun_ref(item, pars)
    C_r = CFun_ref(item, pars)
    K_r = KFun_ref(item, pars)
    theta_ref.append(theta_r)
    C_ref.append(C_r)
    K_ref.append(K_r)


theta = thetaFun(psi, pars)
h = hFun(theta, pars)
C = CFun(psi, pars)
K = KFun(psi, pars)

theta1 = thetaFun1(psi, p, pars)
h1 = hFun1(theta1, p, pars)
C1 = CFun1(psi, p, pars)
K1 = KFun1(psi, p, pars)


theta2 = thetaFun_tanh1(psi, p, pars)
h2 = hFun_tanh1(theta1, p, pars)
C2 = CFun_tanh1(psi, p, pars)
K2 = KFun_tanh1(psi, p, pars)
K3 = hydraulic_conductivity(psi, pars)
K_he = KFun_he(psi, pars)

plt.figure()
# plt.rcParams['figure.figsize'] = (5.0, 10.0)
plt.subplot(411)
plt.plot(psi, theta_ref, 'y--')
plt.plot(psi,theta, 'b:')
plt.plot(psi,theta1,'r:')
plt.plot(psi,theta2,'g:')
plt.ylabel(r'$\theta$', fontsize=20)

plt.subplot(412)
plt.plot(psi,h, 'b:')
plt.plot(psi,h1, 'r:')
plt.plot(psi,h2,'g:')
plt.ylabel(r'$h$', fontsize=20)

plt.subplot(413)
plt.plot(psi, C_ref, 'y--')
plt.plot(psi,C, 'b:')
plt.plot(psi,C1,'r:')
plt.plot(psi,C2,'g:')
plt.ylabel(r'$C$',fontsize=20)

plt.subplot(414)
plt.plot(psi, K_ref, 'y--')
# plt.plot(psi,K, 'b:')
# plt.plot(psi,K1,'r:')
plt.plot(psi,K2,'g:')
plt.plot(psi,K3,'r:')
plt.plot(psi,K_he,'b:')

plt.ylabel(r'$K$', fontsize=20)
plt.xlabel(r'$\psi$', fontsize=20)
plt.show()