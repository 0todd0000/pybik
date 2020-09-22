#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:31:52 2020

@author: ben

WARNING!
This script takes a long time to complete: easily
more than 24 hours, even on a fast PC.
"""
# =============================================================================
# simulations: rotation3
# =============================================================================

from time import time
import numpy as np
import pandas as pd
import pybik

S      = 1000                   # nr of simulations
sigmas = np.array([.2, 1., 2.]) # noise SD levels (mm)

np.random.seed( 0 )
err   = np.zeros( (S*len(sigmas), 12) )
times = np.zeros( (S*len(sigmas), 8) )
k     = 0
for i in range(len(sigmas)):
    for s in range(S):
        
        ## initial marker positions (mm)
        sz = np.random.uniform(75, 150)  # size of markerplate
        d  = np.random.uniform(100, 400) # distance from origin
        r0 = np.array([[sz,0.,d],[-sz,0.,d],[0.,sz,d],[0.,-sz,d]])
        r0 = pybik.MarkerCluster( r0 )

        ## true motion
        theta = np.random.uniform(0, np.pi)
        r     = pybik.random.rotation3( theta )
        r1    = pybik.MarkerCluster( r.apply( r0.toarray() ) )
            
        ## add noise (mm)
        sigma = sigmas[i]
        r0n   = r0.add_noise(sigma)
        r1n   = r1.add_noise(sigma)
            
        ## LS-IK + (V)BIK
        model = pybik.mechanisms.rotation3(r0n, r1n, sigma)
        t0 = time(); r_ls_svd = model.ik_svd();           times[k,0] = time() - t0
        t0 = time(); r_lsquat = model.ik_ls_quat();       times[k,1] = time() - t0
        t0 = time(); r_nlls   = model.ik_nlls();          times[k,2] = time() - t0
        t0 = time(); r_vbik   = model.ik_vb();            times[k,3] = time() - t0
        t0 = time(); r_bik2   = model.ik_bik_axang2()[0]; times[k,4] = time() - t0
        t0 = time(); r_bik3   = model.ik_bik_eulYXY()[0]; times[k,5] = time() - t0
        t0 = time(); r_bik4   = model.ik_bik_eulZXY()[0]; times[k,6] = time() - t0
        t0 = time(); r_bik6   = model.ik_bik_quat2()[0];  times[k,7] = time() - t0
        #unused models
        #r_bik1, sigma_bik1 = model.ik_bik_axang1()
        #r_bik5, sigma_bik5 = model.ik_bik_quat1()
        # r_bik7, sigma_bik7 = model.ik_bik_matrix()
            
        ## estimation errors
        err_ls_svd  = np.degrees( (r_ls_svd * r.inv()).magnitude() )
        err_ls_quat = np.degrees( (r_lsquat * r.inv()).magnitude() )
        err_nlls    = np.degrees( (r_nlls * r.inv()).magnitude() )
        err_vbik    = np.degrees( (r_vbik * r.inv()).magnitude() )
        err_bik2    = np.degrees( (r_bik2 * r.inv()).magnitude() )
        err_bik3    = np.degrees( (r_bik3 * r.inv()).magnitude() )
        err_bik4    = np.degrees( (r_bik4 * r.inv()).magnitude() )
        err_bik6    = np.degrees(  (r_bik6 * r.inv()).magnitude() )
        #err_bik1    = np.degrees( (r_bik1 * r.inv()).magnitude() )
        #err_bik5    = np.degrees( (r_bik5 * r.inv()).magnitude() )
        #err_bik7    = np.degrees( (r_bik7 * r.inv()).magnitude() )
            
        err[k,:] = np.array([sigmas[i], theta, sz, d,
                             err_ls_svd, err_ls_quat, err_nlls, err_vbik,
                             err_bik2, err_bik3, err_bik4, err_bik6])
        k        += 1
        print(i,s)



## save data frames for analysis
errors         = pd.DataFrame(err)
errors.columns = ['sigma','theta','size','distance','SVD','lsQUAT','NLLS',
                  'VBIK','BIKrotvec','BIKeuler','BIKcardan','BIKquat']
errors.to_csv('errors_20092020.csv')

times = pd.DataFrame(times)
times.columns = ['SVD','lsQUAT','NLLS','VBIK',
                 'BIKrotvec','BIKeuler','BIKcardan','BIKquat']
times.to_csv('comptimes_20092020.csv')





