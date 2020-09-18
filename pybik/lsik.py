#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:45:38 2020

@author: ben
"""
# =============================================================================
# least-squares IK
# =============================================================================

from scipy.spatial.transform import Rotation as Rot
import numpy as np
from numpy.linalg import svd, det, eig, inv


def LSsvd(q0n, q1n):
    '''
    LS-IK using Singular Value Decomposition
    reference: Soderkwist and Wedin (1993) Journal of Biomechanics
    '''
    # mean-centered marker positions
    Q0n_cent = q0n.toarray(centered = True).T
    Q1n_cent = q1n.toarray(centered = True).T
    
    # Singular Value Decomposition of relative positions
    Q         = Q1n_cent @ Q0n_cent.T
    U, D, V   = svd(Q)
    r         = Rot.from_matrix( U @ np.diag([1, 1, det(U @ V)]) @ V )
    return r


def LSquat(q0n, q1n):
    '''
    LS-IK using quaternion algebra
    reference: Horn B.K.P. (1987). Closed-form solution of absolute orientation using unit 
    quaternions. Journal of the Optical Society of America, 4(4), 629--642, 1987
    '''
    # mean-centered marker positions
    Q0n_cent = q0n.toarray(centered = True).T
    Q1n_cent = q1n.toarray(centered = True).T
    
    # construction of the 4*4 matrix N
    M          = Q1n_cent @ Q0n_cent.T
    delta      = np.array([ M[1,2]-M[2,1], M[2,0]-M[0,2], M[0,1]-M[1,0] ])
    N          = np.zeros((4,4))
    N[0,:]     = np.array([ np.trace(M), delta[0], delta[1], delta[2] ])
    N[1:4,0]   = delta
    N[1:4,1:4] = M + M.T - np.trace(M)*np.eye(3)
    
    # the quaternion we want is the eigenvector corresponding to the maximal 
    # eigenvalue (these are real valued as N is a real, symmetric matrix)
    eigenv, eigenV = eig(N)
    q              = eigenV[:,eigenv.argmax()]
    q              = np.array([ q[1], q[2], q[3], -q[0] ]) # convert to [xyzw]
    r              = Rot.from_quat( q )
    return r


def nlLS(q0n, q1n, iters = 1000):
    '''
    non-linear LS-IK
    ref: Chapelle et al. (2009). Variational Bayesian inference for a non-
    linear forward model. IEEE Transactions on Signal Processing, 57(1), 223
    Eq. 36
    '''
    y = q1n.toarray().flatten()
    m = LSsvd(q0n, q1n) # initial guess
    for i in range(iters):
        Jac   = np.array( [J(m, x) for x in q0n.toarray()] ).reshape(12,4)
        yhat  = m.apply( q0n.toarray() ).flatten()
        k     = y - yhat
        m_new = Rot.from_quat( 
            m.as_quat() + inv(Jac.T @ Jac) @ Jac.T @ k
            )
        # stop if change in m is < 0.1 degrees
        if ( (m_new * m.inv()).magnitude() < np.radians(0.1) ):
            break
        m = m_new
        
    if (i == iters - 1):
        raise Warning('algorithm reached maxIter')
        
    return m


def sksym(a):
    '''return skew-symmetric matrix (cross-product matrix) of vector a'''
    return np.array([ [0,-a[2],a[1]], [a[2],0,-a[0]], [-a[1],a[0],0] ])

def J(quat, a):
    '''
    define Jacobian matrix for a rotation wrt the quaternion
    ref: Joan Sola (). Quaternion kinematics for the error-state Kalman filter
    Eq. 174
    '''
    w = quat.as_quat()[3]  # scalar part
    v = quat.as_quat()[:3] # vector part
    r = (w*a + np.cross(v,a)).reshape(3,1)
    l = np.dot(v,a) * np.eye(3) + np.outer(v,a) - np.outer(a,v) - w * sksym(a)
    J = 2 * np.concatenate((l, r), axis = 1)
    return J
        
