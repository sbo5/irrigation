from __future__ import print_function, division # Grab some handy Python3 stuff.
import scipy.linalg
import casadi
import numpy as np
from contextlib import contextmanager
import mpctools as mpc



#######################################################################################
def ekf_continuous_discrete(f,h,x_hat,u,w,y,P,Q,R,Nw,Delta,f_jacx=None,f_jacw=None,h_jacx=None):
    """
    This ekf is discrete-time to continuous-time EKF. That is, it can be used to 
    ODE models. 
    
    f and h should be casadi functions. f must be discrete-time. P, Q, and R
    are the prior, state disturbance, and measurement noise covariances. Note
    that f must be f(x,u,w) and h must be h(x).
    
    
    The value of x that should be fed is xhat(k | k-1), and the value of P
    should be P(k | k-1). xhat will be updated to xhat(k | k) and then advanced
    to xhat(k+1 | k), while P will be updated to P(k | k) and then advanced to
    P(k+1 | k). The return values are a list as follows
    
        [P(k+1 | k), xhat(k+1 | k), P(k | k), xhat(k | k)]
        
    Depending on your specific application, you will only be interested in
    some of these values.
    """
    # Check jacobians.
    if f_jacx is None:
        f_jacx = f.jacobian(0)
    if f_jacw is None:
        f_jacw = f.jacobian(2)
    if h_jacx is None:
        h_jacx = h.jacobian(0)

    
    P_k_kminus = P                         # Pmp1_cov is the covariance matrix P(k|k-1)
    xekf_k_kminus = x_hat                 # This is the estimate xhat(k | k-1)
    C = np.array(h_jacx(xekf_k_kminus)[0])   # Linearize the output equations to obtain C matrix
    
    # update step
    Lk = mpc.mtimes( P_k_kminus, np.transpose(C), np.linalg.inv( np.dot(np.dot(C,P_k_kminus),np.transpose(C))+R))  # L = Pc(:,:,i)*C'/(C*Pc(:,:,i)*C'+Rc)
    xekf = xekf_k_kminus + np.dot( Lk,(y - np.dot(C,xekf_k_kminus)) )         ## update current estimate xhat(k|k)
    P_cov = P_k_kminus - mpc.mtimes(Lk, C, P_k_kminus)     ## update current estimate of P(k|k) based on P(k|k-1)
        
    # predict step
    A = np.array(f_jacx(xekf, u, np.zeros((Nw,1)) )[0])
    B = np.array(f_jacw(xekf, u, np.zeros((Nw,1)) )[0])
    
    [Ad, Bd] = c2d(A, B, Delta, Bp=None, f=None, asdict=False)
    P_kplus_k = mpc.mtimes( Ad, P_cov, np.transpose(Ad) ) + Q      # update matrix P (k+1|k)
    
    return [P_kplus_k, P, xekf] 


#######################################################################################


#######################################################################################
def ekf_discrete_discrete(f,h,x,u,w,y,P,Q,R,f_jacx=None,f_jacw=None,h_jacx=None):
    """
    This ekf is discrete-time to discrete-time EKF. That is, it can be used to 
    discrete-time state-space model. For ODEs, please use "ekf_continuous_discrete"
    
    Updates the prior distribution P^- using the Extended Kalman filter.
    
    f and h should be casadi functions. f must be discrete-time. P, Q, and R
    are the prior, state disturbance, and measurement noise covariances. Note
    that f must be f(x,u,w) and h must be h(x).
    
    If specified, f_jac and h_jac should be initialized jacobians. This saves
    some time if you're going to be calling this many times in a row, althouth
    it's really not noticable unless the models are very large.
    
    The value of x that should be fed is xhat(k | k-1), and the value of P
    should be P(k | k-1). xhat will be updated to xhat(k | k) and then advanced
    to xhat(k+1 | k), while P will be updated to P(k | k) and then advanced to
    P(k+1 | k). The return values are a list as follows
    
        [P(k+1 | k), xhat(k+1 | k), P(k | k), xhat(k | k)]
        
    Depending on your specific application, you will only be interested in
    some of these values.
    """
    
    # Check jacobians.
    if f_jacx is None:
        f_jacx = f.jacobian(0)
    if f_jacw is None:
        f_jacw = f.jacobian(2)
    if h_jacx is None:
        h_jacx = h.jacobian(0)
        
    # Get linearization of measurement.
    C = np.array(h_jacx(x)[0])
    yhat = np.array(h(x)[0]).flatten()
    
    # Advance from x(k | k-1) to x(k | k).
    xhatm = x                                          # This is xhat(k | k-1)    
    Pm = P                                             # This is P(k | k-1)    
    L = scipy.linalg.solve(C.dot(Pm).dot(C.T) + R, C.dot(Pm)).T          
    xhat = xhatm + L.dot(y - yhat)                     # This is xhat(k | k) 
    P = (np.eye(Pm.shape[0]) - L.dot(C)).dot(Pm)       # This is P(k | k)
    
    # Now linearize the model at xhat.
    w = np.zeros(w.shape)
    A = np.array(f_jacx(xhat, u, w)[0])
    G = np.array(f_jacw(xhat, u, w)[0])
    
    # Advance.
    Pmp1 = A.dot(P).dot(A.T) + G.dot(Q).dot(G.T)       # This is P(k+1 | k)
    xhatmp1 = np.array(f(xhat, u, w)).flatten()     # This is xhat(k+1 | k)    
    
    return [Pmp1, xhatmp1, P, xhat]
#######################################################################################
    

#######################################################################################
def c2d(A, B, Delta, Bp=None, f=None, asdict=False):
#    """
#    Discretizes affine system (A, B, Bp, f) with timestep Delta.
#    
#    This includes disturbances and a potentially nonzero steady-state, although
#    Bp and f can be omitted if they are not present.
#    
#    If asdict=True, return value will be a dictionary with entries A, B, Bp,
#    and f. Otherwise, the return value will be a 4-element list [A, B, Bp, f]
#    if Bp and f are provided, otherwise a 2-element list [A, B].
#    """
    n_A = A.shape[0]
    I = np.eye(n_A)
    D = scipy.linalg.expm(Delta*np.vstack((np.hstack([A, I]),
                                             np.zeros((n_A, 2*n_A)))))
    Ad = D[:n_A,:n_A]
    Id = D[:n_A,n_A:]
    Bd = Id.dot(B)
    Bpd = None if Bp is None else Id.dot(Bp)
    fd = None if f is None else Id.dot(f)   
            
    if asdict:
        retval = dict(A=Ad, B=Bd, Bp=Bpd, f=fd)
    elif Bp is None and f is None:
        retval = [Ad, Bd]
    else:
        retval = [Ad, Bd, Bpd, fd]
    return retval
#######################################################################################