#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:03:23 2020

@author: ben
"""

from scipy.spatial.transform import Rotation as Rot
from . import directional as dr
from . import lsik
from . import vbik
import pymc as pm
import numpy as np
from math import pi, sin, cos, atan2, acos


# =============================================================================
# rotation through 3 angles around origin
# =============================================================================
class rotation3():
    
    def __init__(self, q0n, q1n, sigma):
        self.q0n   = q0n
        self.q1n   = q1n
        self.sigma = sigma
    
    
    # ========== #
    ''' LS-IK  '''
    # ========== #
    def ik_svd(self):
        r_ls_svd = lsik.LSsvd(self.q0n, self.q1n)
        return r_ls_svd
    
    def ik_ls_quat(self):
        r_ls_quat = lsik.LSquat(self.q0n, self.q1n) 
        return r_ls_quat
    
    def ik_nlls(self):
        r_nlls = lsik.nlLS(self.q0n, self.q1n)
        return r_nlls
    
    
    # ========================= #
    ''' Variational Bayes-IK  ''' 
    # ========================= #
    def ik_vb(self):
        r_vb = vbik.VB(self.q0n, self.q1n)
        return r_vb
    
    
    # ================= #
    ''' MCMC MODEL 1  '''
    # ================= #
    def ik_bik_axang1(self, niter = 1E5, nburn = 5E4, nthin = 5, aMH = True):
        
        q0n = self.q0n.toarray()
        q1n = self.q1n.toarray()
        
        # prior on measurement error (precision)
        sigma = self.sigma
        tau   = pm.Gamma('tau', alpha = 1/3, beta = 1/3, value = 1/sigma**2)
        
        # priors on rotation
        r0    = self.ik_svd() # initial guess
        r00   = r0.as_rotvec()/r0.magnitude()
        phi0  = atan2(r00[1], r00[0])
        psi0  = acos(r00[2])
        phi   = pm.Uniform('phi', -pi, pi, value = phi0)  # azimuthal angle (axis)
        psi   = pm.Uniform('psi', 0, pi, value = psi0)    # polar angle (axis)
        theta = pm.Uniform('theta', 0, pi, value = r0.magnitude())  # angle
                
        # forward kinematics model
        @pm.deterministic
        def fwk(phi = phi, psi = psi, theta = theta):
            u     = np.array([sin(psi)*cos(phi), sin(psi)*sin(phi), cos(psi)])
            R     = Rot.from_rotvec(theta * u)
            q1hat = R.apply(q0n)
            return q1hat.flatten()
        
        # data likelihood
        lik = pm.Normal('lik', mu = fwk, tau = tau, 
                        value = q1n.flatten(), observed = True)
        
        # MCMC sampling + posterior point estimation
        mcmc = pm.MCMC({'tau':tau, 'phi':phi, 'psi':psi, 'theta':theta, 'lik':lik})
        if aMH == True:
            mcmc.use_step_method(pm.AdaptiveMetropolis,
                                 [mcmc.tau, mcmc.phi, mcmc.psi, mcmc.theta],
                                 interval = 5000, delay = 10000)
        mcmc.sample(iter = niter + nburn, burn = nburn, thin = nthin)
        tau       = mcmc.trace('tau')[:]
        sigma     = 1/np.sqrt(tau)
        sigma_bik = sigma.mean()
        phi       = mcmc.trace('phi')[:].mean()
        psi       = mcmc.trace('psi')[:].mean()
        theta     = mcmc.trace('theta')[:].mean()
        u         = np.array([sin(psi)*cos(phi), sin(psi)*sin(phi), cos(psi)])
        r_bik     = Rot.from_rotvec(theta * u)
        
        return r_bik, sigma_bik
    
    
    # ================= #
    ''' MCMC MODEL 2  '''
    # ================= #
    def ik_bik_axang2(self, niter = 1E5, nburn = 5E4, nthin = 5, aMH = True):
        
        q0n = self.q0n.toarray()
        q1n = self.q1n.toarray()
        
        # prior on measurement error (precision)
        sigma = self.sigma
        tau   = pm.Gamma('tau', alpha = 1/3, beta = 1/3, value = 1/sigma**2)
        
        # priors on rotation
        r0    = self.ik_svd() # initial guess
        r00   = r0.as_rotvec()/r0.magnitude()
        u     = pm.Normal('u', 0, 1, size = 3, value = r00)         # axis
        theta = pm.Uniform('theta', 0, pi, value = r0.magnitude())  # angle
                
        # forward kinematics model
        @pm.deterministic
        def fwk(u = u, theta = theta):
            u     = u/np.linalg.norm(u)
            R     = Rot.from_rotvec(theta * u)
            q1hat = R.apply(q0n)
            return q1hat.flatten()
        
        # data likelihood
        lik = pm.Normal('lik', mu = fwk, tau = tau, 
                        value = q1n.flatten(), observed = True)
        
        # MCMC sampling + posterior point estimation
        mcmc = pm.MCMC({'tau':tau, 'u':u, 'theta':theta, 'lik':lik})
        if aMH == True:
            mcmc.use_step_method(pm.AdaptiveMetropolis,
                                 [mcmc.tau, mcmc.u, mcmc.theta],
                                 interval = 5000, delay = 10000)
        mcmc.sample(iter = niter + nburn, burn = nburn, thin = nthin)
        tau       = mcmc.trace('tau')[:]
        sigma     = 1/np.sqrt(tau)
        sigma_bik = sigma.mean()
        u         = mcmc.trace('u')[:].mean(axis = 0)
        u         = u/np.linalg.norm(u)
        theta     = mcmc.trace('theta')[:].mean()
        r_bik     = Rot.from_rotvec(theta * u)
        
        return r_bik, sigma_bik
    
    
    # ================= #
    ''' MCMC MODEL 3  '''
    # ================= #
    def ik_bik_eulYXY(self, niter = 1E5, nburn = 5E4, nthin = 5, aMH = True):
        q0n = self.q0n.toarray()
        q1n = self.q1n.toarray()
        
        # prior on measurement error (precision)
        sigma = self.sigma
        tau   = pm.Gamma('tau', alpha = 1/3, beta = 1/3, value = 1/sigma**2)
        
        # prior on rotation (Euler/Cardan angles)
        r0                    = self.ik_svd() # initial guess
        alpha0, beta0, gamma0 = r0.as_euler('YXY')
        alpha = pm.Uniform('alpha', -pi, pi, value = alpha0) # first angle
        beta  = pm.Uniform('beta', 0, pi, value = beta0)     # second angle
        gamma = pm.Uniform('gamma', -pi, pi, value = gamma0) # third angle
        
        # forward kinematics model
        @pm.deterministic
        def fwk(alpha = alpha, beta = beta, gamma = gamma):
            R     = Rot.from_euler('YXY', np.array([alpha, beta, gamma]))
            q1hat = R.apply(q0n)
            return q1hat.flatten()
        
        # data likelihood
        lik = pm.Normal('lik', mu = fwk, tau = tau, 
                        value = q1n.flatten(), observed = True)
        
        # MCMC sampling + posterior point estimation
        mcmc = pm.MCMC({'tau':tau, 'alpha':alpha, 'beta':beta, 'gamma':gamma, 'lik':lik})
        if aMH == True:
            mcmc.use_step_method(pm.AdaptiveMetropolis,
                                 [mcmc.tau, mcmc.alpha, mcmc.beta, mcmc.gamma],
                                 interval = 5000, delay = 10000)
        mcmc.sample(iter = niter + nburn, burn = nburn, thin = nthin)
        tau       = mcmc.trace('tau')[:]
        sigma     = 1/np.sqrt(tau)
        sigma_bik = sigma.mean()
        alpha     = mcmc.trace('alpha')[:].mean()
        beta      = mcmc.trace('beta')[:].mean()
        gamma     = mcmc.trace('gamma')[:].mean()
        r_bik     = Rot.from_euler('YXY', np.array([alpha, beta, gamma]))
        
        return r_bik, sigma_bik
    
    
    # ================= #
    ''' MCMC MODEL 4  '''
    # ================= #
    def ik_bik_eulZXY(self, niter = 1E5, nburn = 5E4, nthin = 5, aMH = True):
        q0n = self.q0n.toarray()
        q1n = self.q1n.toarray()
        
        # prior on measurement error (precision)
        sigma = self.sigma
        tau   = pm.Gamma('tau', alpha = 1/3, beta = 1/3, value = 1/sigma**2)
        
        # prior on rotation (Euler/Cardan angles)
        r0                    = self.ik_svd() # initial guess
        alpha0, beta0, gamma0 = r0.as_euler('ZXY')
        alpha = pm.Uniform('alpha', -pi, pi, value = alpha0)    # first angle
        beta  = pm.Uniform('beta', -pi/2, pi/2, value = beta0)  # second angle
        gamma = pm.Uniform('gamma', -pi, pi, value = gamma0)    # third angle
        
        # forward kinematics model
        @pm.deterministic
        def fwk(alpha = alpha, beta = beta, gamma = gamma):
            R     = Rot.from_euler('ZXY', np.array([alpha, beta, gamma]))
            q1hat = R.apply(q0n)
            return q1hat.flatten()
        
        # data likelihood
        lik = pm.Normal('lik', mu = fwk, tau = tau, 
                        value = q1n.flatten(), observed = True)
        
        # MCMC sampling + posterior point estimation
        mcmc = pm.MCMC({'tau':tau, 'alpha':alpha, 'beta':beta, 'gamma':gamma, 'lik':lik})
        if aMH == True:
            mcmc.use_step_method(pm.AdaptiveMetropolis,
                                 [mcmc.tau, mcmc.alpha, mcmc.beta, mcmc.gamma],
                                 interval = 5000, delay = 10000)
        mcmc.sample(iter = niter + nburn, burn = nburn, thin = nthin)
        tau       = mcmc.trace('tau')[:]
        sigma     = 1/np.sqrt(tau)
        sigma_bik = sigma.mean()
        alpha     = mcmc.trace('alpha')[:].mean()
        beta      = mcmc.trace('beta')[:].mean()
        gamma     = mcmc.trace('gamma')[:].mean()
        r_bik     = Rot.from_euler('ZXY', np.array([alpha, beta, gamma]))
        
        return r_bik, sigma_bik
    
    
    # ================= #
    ''' MCMC MODEL 5  '''
    # ================= #
    def ik_bik_quat1(self, niter = 1E5, nburn = 5E4, nthin = 5, aMH = False):
        q0n = self.q0n.toarray()
        q1n = self.q1n.toarray()
        
        # prior on measurement error (precision)
        sigma = self.sigma
        tau   = pm.Gamma('tau', alpha = 1/3, beta = 1/3, value = 1/sigma**2)
        
        # prior on rotation (quaternion)
        r0 = self.ik_svd() # initial guess
        @pm.stochastic(dtype = float)
        def r(mu    = np.array([1.,0.,0.,0.]), 
              kappa = 1E-100, 
              value = r0.as_quat()):
            def logp(value, mu, kappa):
                return dr.vmf_density(value, mu, kappa, logdens = True)
            def random(mu, kappa):
                return dr.rvmf(n = 1, mu = mu, kappa = kappa)
        
        # forward kinematics model
        @pm.deterministic
        def fwk(r = r):
            R     = Rot.from_quat(r)
            q1hat = R.apply(q0n)
            return q1hat.flatten()
        
        # data likelihood
        lik = pm.Normal('lik', mu = fwk, tau = tau, 
                        value = q1n.flatten(), observed = True)
        
        # MCMC sampling + posterior point estimation
        mcmc = pm.MCMC({'tau':tau, 'r':r, 'lik':lik})
        if aMH == True:
            mcmc.use_step_method(pm.AdaptiveMetropolis,
                                 [mcmc.tau, mcmc.r],
                                 interval = 5000, delay = 10000)
        mcmc.sample(iter = niter + nburn, burn = nburn, thin = nthin)
        tau       = mcmc.trace('tau')[:]
        sigma     = 1/np.sqrt(tau)
        sigma_bik = sigma.mean()
        r         = Rot.from_quat( mcmc.trace('r')[:] )
        r_bik     = r.mean()
        
        return r_bik, sigma_bik
    
    
    # ================= #
    ''' MCMC MODEL 6  '''
    # ================= #
    def ik_bik_quat2(self, niter = 1E5, nburn = 5E4, nthin = 5, aMH = False):
        q0n = self.q0n.toarray()
        q1n = self.q1n.toarray()
        
        # prior on measurement error (precision)
        sigma = self.sigma
        tau   = pm.Gamma('tau', alpha = 1/3, beta = 1/3, value = 1/sigma**2)
        
        # prior on rotation (quaternion)
        r0 = self.ik_svd() # initial guess
        r  = pm.Normal('r', 0, 1, size = 4, value = r0.as_quat())
        
        # forward kinematics model
        @pm.deterministic
        def fwk(r = r):
            R     = Rot.from_quat(r)
            q1hat = R.apply(q0n)
            return q1hat.flatten()
        
        # data likelihood
        lik = pm.Normal('lik', mu = fwk, tau = tau, 
                        value = q1n.flatten(), observed = True)
        
        # MCMC sampling + posterior point estimation
        mcmc = pm.MCMC({'tau':tau, 'r':r, 'lik':lik})
        if aMH == True:
            mcmc.use_step_method(pm.AdaptiveMetropolis,
                                 [mcmc.tau, mcmc.r],
                                 interval = 5000, delay = 10000)
        mcmc.sample(iter = niter + nburn, burn = nburn, thin = nthin)
        tau       = mcmc.trace('tau')[:]
        sigma     = 1/np.sqrt(tau)
        sigma_bik = sigma.mean()
        r         = Rot.from_quat( mcmc.trace('r')[:] )
        r_bik     = r.mean()
        
        return r_bik, sigma_bik
    
    
    # ================= #
    ''' MCMC MODEL 7  ''' # not ready yet: no proper prior distribution !
    # ================= #
    # def ik_bik_matrix(self, niter = 1E5, nburn = 5E4, nthin = 5):
    #     q0n = self.q0n.toarray()
    #     q1n = self.q1n.toarray()
        
    #     # prior on measurement error (precision)
    #     sigma = self.sigma
    #     tau   = pm.Gamma('tau', alpha = 1/3, beta = 1/3, value = 1/sigma**2)
        
    #     # prior on rotation matrix
    #     r0 = self.ik_svd() # initial guess
    #     @pm.stochastic(dtype = float)
    #     def R(F = np.eye(3), value = r0.as_matrix()):
    #         def logp(value, F):
    #             return dr.matrixvmf_density(value, F, logdens = True)
    #         def random(F):
    #             return dr.habeck_rot(F)
        
    #     # forward kinematics model
    #     @pm.deterministic
    #     def fwk(R = R):
    #         R     = Rot.from_matrix(R)
    #         q1hat = R.apply(q0n)
    #         return q1hat.flatten()
        
    #     # data likelihood
    #     lik = pm.Normal('lik', mu = fwk, tau = tau, 
    #                     value = q1n.flatten(), observed = True)
        
    #     # MCMC sampling + posterior point estimation
    #     mcmc      = pm.MCMC([tau, R, lik])
    #     mcmc.sample(iter = niter + nburn, 
    #                 burn = nburn, 
    #                 thin = nthin)
    #     tau       = mcmc.trace('tau')[:]
    #     sigma     = 1/np.sqrt(tau)
    #     sigma_bik = sigma.mean()
    #     r         = Rot.from_matrix( mcmc.trace('R')[:] )
    #     r_bik     = r.mean()
        
    #     return r_bik, sigma_bik



# =============================================================================
# rotation around origin + translation 
# =============================================================================
'''to do'''
class rottrans3():
    
    def __init__(self, q0n, q1n, sigma):
        self.q0n   = q0n
        self.q1n   = q1n
        self.sigma = sigma
        

# =============================================================================
# rotation around arbitrary point S that is not the origin
# =============================================================================
'''to do'''
class rotation3_aroundS():
    
    def __init__(self, q0n, q1n, sigma):
        self.q0n   = q0n
        self.q1n   = q1n
        self.sigma = sigma
        

# =============================================================================
# rotation around arbitrary point + translation
# =============================================================================
'''to do'''
class rottrans3_aroundS():
    
    def __init__(self, q0n, q1n, sigma):
        self.q0n   = q0n
        self.q1n   = q1n
        self.sigma = sigma
    