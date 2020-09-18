#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:36:37 2020

@author: ben
"""

from scipy.spatial.transform import Rotation as Rot
from . import directional as dr


# =============================================================================
# rotation through 3 random angles around origin
# =============================================================================
def rotation3(theta, n = 1, mu = [1., 0., 0.], kappa = 0.):
    u      = dr.rvmf(n, mu, kappa)
    return Rot.from_rotvec(theta * u)



