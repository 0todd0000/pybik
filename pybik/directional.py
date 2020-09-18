'''
Python translations of key functions from the R package "Directional"
https://cran.r-project.org/web/packages/Directional/
@author: Todd Pataky
'''

from math import log, exp, pi, atan2, acos, asin, sin, cos, sqrt
import numpy as np
from scipy import special
import random



def _shape(x):
	if x.ndim==1:
		n,p = x.size, 1
	else:
		n,p = x.shape  #number of observations, number of DVs
	return n,p



def rotation(a, b):
	'''
	a and b are two unit vectors
	Calculates the rotation matrix
	to move a to b along the geodesic path
	on the unit sphere which connects a to b
	'''
	p     = a.size
	ab    = (a * b).sum()
	ca    = a - b * ab
	ca    = ca / np.linalg.norm(ca)
	A     = np.matrix(b).T * np.matrix(ca)
	A    -= A.T
	theta = acos( ab )
	
	bm    = np.matrix(b).T
	cam   = np.matrix(ca).T
	B0    = np.eye(p) + sin(theta) * A
	B1    = (cos(theta) - 1) * (bm*bm.T + cam*cam.T )
	return B0 + B1




def rvmf(n = 1, mu = [1., 0., 0.], kappa = 1.):
	'''
	Translation of "rvmf" from the Directional package for R
	n     : sample size
	mu    : mean direction
	kappa : concentration parameter
	Outputs:
	x  : (J,I) array of random unit vectors from a Von Mises - Fisher distribution
	'''
	mu       = np.asarray(mu, dtype=float)
	d        = mu.size
	mu      /= np.linalg.norm(mu)
	ini      = np.zeros(d)
	ini[-1]  = 1
	d1       = d - 1
	v1       = np.random.randn( n , d1 )
	v        = (v1.T / np.linalg.norm(v1, axis=1)).T
	b        = ( -2 * kappa + sqrt(4 * kappa**2 + d1**2) ) / d1
	x0       = (1-b) / (1+b)
	w        = np.zeros(n)
	m        = 0.5 * d1
	ca       = kappa * x0 + (d - 1) * log(1 - x0**2)
	for i in range(n):
		ta   = -1000
		u    = 1
		while ( ta - ca < log(u) ):
			# use numpy random functions so that seed can be controlled using np.random.seed
			z    = np.random.beta(m, m)
			u    = np.random.rand()
			w[i] = ( 1 - (1 + b) * z ) / ( 1 - (1 - b) * z )
			ta   = kappa * w[i] + d1 * log(1 - x0 * w[i])
	S = np.vstack([(np.sqrt(1 - w**2) * v.T), w]).T
	A = rotation(mu, ini)
	x = (S @ A)
	return x



def vmf_density(x, mu, kappa, logdens = False):
    '''
	density for a von-Mises Fisher distribution with mean direction mu and concentration kappa
	Adapted from vmf.density.R, written by Tsagris Michail 10/2013 (mtsagris@yahoo.gr)
	as available in the "Directional" package for R
	'''
    x       = np.matrix(x)
    n, p    = _shape(x)
    if p == 1: x = x.T
    n, p    = _shape(x)
    besselI = lambda a,b: special.ive(b,a)
    den     = (p/2 - 1) * log(kappa) - 0.5 * p * log(2 * pi) \
        + kappa * mu @ x.T - log(besselI(kappa, p/2 - 1)) - kappa
    if logdens == False:
        den = exp(den)
    return den



def habeck_rot(F):
	'''
	Generate a random 3D rotation matrix
	'''
	u,d,v  = np.linalg.svd(F)
	U,D,tV = [np.matrix(x) for x in (u,d,v)]
	D      = D.T

	if np.linalg.det(U*tV) < 0:
		U[:,2] *= -1
		D[2]   *= -1

	lamda1,lamda2,lamda3 = np.array(D).flatten()

	### generate Euler angles:
	Beta_val  = 0
	kappa_phi = (lamda1 + lamda2) * ( cos(Beta_val/2) ) ** 2
	kappa_shi = (lamda1 - lamda2) * ( sin(Beta_val/2) ) ** 2

	phi       = random.uniform(0, 2*pi) if kappa_phi == 0 else random.vonmisesvariate(0, kappa_phi)
	shi       = random.uniform(0, 2*pi) if kappa_shi == 0 else random.vonmisesvariate(0, kappa_shi)
	u         = float( np.random.binomial(1, 0.5, size=1) )
	alpha     = 0.5 * (phi + shi) + pi * u
	gamma     = 0.5 * (phi - shi) + pi * u


	kappa_Beta = (lamda1 + lamda2) * cos(phi) + (lamda1 - lamda2) * cos(shi) + 2 * lamda3
	r          = random.uniform(0, 1)
	x          = 1 + 2 * log( r + (1 - r) * exp(-kappa_Beta) ) / kappa_Beta
	Beta_val   = acos(x)

	### build rotation matrix:
	a11 = cos(alpha) * cos(Beta_val) * cos(gamma) - sin(alpha) * sin(gamma)
	a21 =  -cos(alpha) * cos(Beta_val) * sin(gamma) - sin(alpha) * cos(gamma)
	a31 = cos(alpha) * sin(Beta_val)
	a12 = sin(alpha) * cos(Beta_val) * cos(gamma) + cos(alpha) * sin(gamma)
	a22 =  -sin(alpha) * cos(Beta_val) * sin(gamma) + cos(alpha) * cos(gamma)
	a32 = sin(alpha) * sin(Beta_val)
	a13 =  -sin(Beta_val) * cos(gamma)
	a23 = sin(Beta_val) * sin(gamma)
	a33 = cos(Beta_val)
	S   = np.matrix( [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]] )
	R   = U * S * tV
	return R


def matrixvmf_density(X, F, logdens = False):
    '''unnormalized density of the Matrix-vonMises-Fisher distribution'''
    if logdens == True:
        den = np.trace(F.T @ X)
    else:
        den = exp(np.trace(F.T @ X))
    return den
