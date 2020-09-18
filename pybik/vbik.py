#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:26:30 2020

@author: ben
"""
# =============================================================================
# variational Bayes IK
# =============================================================================


from scipy.spatial.transform import Rotation as Rot
from scipy.special import digamma, loggamma
from numpy.linalg import inv, cholesky
from numpy import log
import numpy as np
from . import lsik


def logdet(X):
    '''function with higher accuracy than np.log(np.linalg.det(X))'''
    logdetX = 2 * np.trace(log( cholesky(X) ))
    return logdetX

def sksym(a):
    '''return skew-symmetric matrix (cross-product matrix) of vector a'''
    return np.array([ [0,-a[2],a[1]], [a[2],0,-a[0]], [-a[1],a[0],0] ])

def J(quat, a):
    '''
    define Jacobian matrix for a rotation wrt the quaternion
    ref: Joan Sola (2017). Quaternion kinematics for the error-state Kalman filter
    Eq. 174
    '''
    w = quat.as_quat()[3]  # scalar part
    v = quat.as_quat()[:3] # vector part
    r = (w*a + np.cross(v,a)).reshape(3,1)
    l = np.dot(v,a) * np.eye(3) + np.outer(v,a) - np.outer(a,v) - w * sksym(a)
    J = 2 * np.concatenate((l, r), axis = 1)
    return J
        

def VB(q0n, q1n, iters = 1000):
    '''Variational Bayes method for inverse kinematics'''
    
    y = q1n.toarray().flatten()
    N = len(y)
    
    ## priors ##
    # quaternion: MVN(mean, precision)  
    m0      = lsik.LSsvd(q0n, q1n) # for the mean we take the SVD estimate
    Lambda0 = 1E-10 * np.eye(4)    # small precision reflects prior ignorance 
    # measurement precision: Gamma(shape, scale)
    s0 = 1/3
    c0 = 1/3
    
    ## VB-EM algorithm ##
    m  = m0
    s  = s0
    c  = N/2 + c0
    F  = - np.finfo('d').max
    for i in range(iters):
        Jac        = np.array( [J(m, x) for x in q0n.toarray()] ).reshape(12,4)
        yhat       = m.apply( q0n.toarray() ).flatten()
        k          = y - yhat
        
        # update equations of the parameters
        Lambda_new = s * c * Jac.T @ Jac + Lambda0
        m_new      = Rot.from_quat( 
            inv(Lambda_new) @ (s * c * Jac.T @ (k + Jac @ m.as_quat()) + Lambda0 @ m0.as_quat())
            )
        s_new = 1 / (1/s0 + 0.5 * (k.T @ k + np.trace(inv(Lambda_new) @ Jac.T @ Jac)))
        
        # free energy
        F_new = - (s_new * c)/s0 \
            + (N/2 + c0 - 1) * (log(s) + digamma(c)) \
                - 0.5 * ( (m_new.as_quat() - m0.as_quat()).T @ Lambda0 @ (m_new.as_quat() - m0.as_quat()) ) \
                    - 0.5 * np.trace( inv(Lambda_new) @ Lambda0  ) \
                        - 0.5 * (k.T @ k + np.trace( inv(Lambda_new) @ Jac.T @ Jac )) \
                                 - s_new * log(c) \
                                     - loggamma(c) \
                                         - c + (N/2 + c - 1) * (log(s) + digamma(c)) \
                                             + 0.5 * logdet(Lambda_new)
        
        # convergence of free energy?
        # stop if change in variational bound is < 0.001%
        if (abs(F - F_new) < abs(0.00001 * F_new)):
            break
        # alternative: ?stop if change in m is < 0.1 degrees?
        # if ( (m_new * m.inv()).magnitude() < np.radians(0.1) ):
        #     break
        F  = F_new
        m  = m_new
        s  = s_new
    
    # raise warning if EM has not converged on maxIter
    if (i == iters - 1):
        raise Warning('algorithm reached maxIter')
        
    return m

