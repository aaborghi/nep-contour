"""
@author: aaborghi
In this code we benchmark the contour integral eigensolvers against the delay 
eigenvalue problem of bigdae.py.
"""

from numpy import array, zeros, concatenate, negative, exp, diag, diagonal
from scipy.special import lambertw
import numpy as np
from matplotlib import pyplot as plt
from src.nlevpfunc import contour_solver

"""Parameters"""
n = 50 #size of the NEP
e0 = np.geomspace(-(10**10),-(10**(-4)),50)
E0 = np.diag(e0)
R = 0.2 #radius of the circle
mu = 0 #center of the circle
V = np.identity(n) #initial projection matrix 

#change of variables function and its derivative
def phi(x):
    return mu + R * np.exp(1j * x)
def dphi(x):
    return 1j * R * np.exp((x * 1j))

#description of the NEP (in this case we have an delay eigenvalue problem)
def T(z):
    return z*np.eye(n) + 0.015*np.exp(-z*8)*np.eye(n) - E0

#Lambert-W function used to get the actual solution of the NEP (so to have a comparison)
def lambertW(ktimes,tau,T1,T0):
    Lamb = array(zeros(n))
    for j in range(-ktimes,ktimes):
            Wk = lambertw(tau * negative(diagonal(T1)) * exp(tau * diagonal(T0)), k=j)
            Wk = array(Wk)
            Sk = (diag(Wk)/tau) - T0
            Lambk = Sk
            Lambk = diagonal(Lambk)
            Lamb = concatenate((Lamb,Lambk), axis=0)
            lenLamb = len(Lamb)
    Lamb = Lamb[n:lenLamb]
    Lamb = Lamb[np.isfinite(Lamb)] #removing all the Nan and Inf
    return Lamb

#Function to find the maximum error
def maxError(eigs, lamb):
    Emin = np.zeros((1,lamb.shape[0]), dtype = 'float')
    for j in range (0, lamb.shape[0]):
        if lamb.shape[0] >= eigs.shape[0]:
            err = np.zeros((1,lamb.shape[0]), dtype = 'float') 
        else: 
            err = np.zeros((1,eigs.shape[0]), dtype = 'float')
        for i in range (0, eigs.shape[0]):
            err[0,i] = np.absolute(eigs[i] - lamb[j]) 
        Emin[0,j] = np.amin(err)            
    return np.max(Emin)


sol = contour_solver(n, T)
l = lambertW(10,8,0.015*np.eye(n),-E0) #setup the lamberW function solver

## Benchmark function
"""
Here we compute the maximum absolute and relative error of the results of each
contour integral eigensolver for an increasing number of quadrature points
"""
def bench(quadNodes):
    np.random.seed(2022) 
    q = 11
    
    
    lambH, _, eigvH = sol.hankel(quadNodes, 40, q, 3, phi, dphi, V)
    
    lambL, _, eigv = sol.single_loewner(quadNodes, 40, q, 3, phi, dphi, 1/2)
    
    nint = 11
    muq = 2 * R * np.exp(1j * np.linspace(-1, 1, num=nint)) + mu 
    lambdaq = -2 * R * np.exp(1j * np.linspace(-1, 1, num=nint)) + mu 
    lvec = np.random.rand(n, nint) + np.random.rand(n, nint) * 1j
    rvec = np.random.rand(n, nint) + np.random.rand(n, nint) * 1j
    
    lambL2, _, eigv2 = sol.multi_loewner(quadNodes, q, phi, dphi, muq, lambdaq, rvec, lvec)

    lambL3, _, eigv3 = sol.multi1side_loewner(quadNodes, q, phi, dphi, muq, rvec, lvec)

    ## ERRORS OF THE APPROXIMATIONS
    Emax = maxError(l, lambH)
    Emax1 = maxError(l, lambL)
    Emax2 = maxError(l, lambL2)    
    Emax3 = maxError(l, lambL3)
    TlambvH, _ = sol.residualError(lambH, eigvH)
    Tlambv, _ = sol.residualError(lambL, eigv)
    Tlambv2, _ = sol.residualError(lambL2, eigv2)
    Tlambv3, _ = sol.residualError(lambL3, eigv3)

    return Emax, Emax1, Emax2, Emax3, TlambvH, Tlambv, Tlambv2, Tlambv3

iterations = 10
Eabs = np.zeros((4,iterations))
Tlambv = np.zeros((4,iterations))
mean = np.zeros((4,iterations))
quadnodes = np.zeros((1,iterations), dtype='int')
for i in range(0,iterations):
    quadnodes[0,i] = (2**i+10)
    Eabs[0,i], Eabs[1,i], Eabs[2,i], Eabs[3,i], Tlambv[0,i], Tlambv[1,i], Tlambv[2,i], Tlambv[3,i] = bench(quadnodes[0,i])


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(quadnodes.ravel(), Eabs[0,:], 'r--', quadnodes.ravel(), Eabs[1,:], 'g-.', quadnodes.ravel(), Eabs[2,:], 'b-o', quadnodes.ravel(), Eabs[3,:], 'y-+') 
ax1.legend([r'Hankel', r'1Loew', r'multiLoew', r'singlemultiLoew'], loc='upper right')
ax1.set(ylabel=r'maximum error')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax2.plot(quadnodes.ravel(), Tlambv[0,:], 'r--', quadnodes.ravel(), Tlambv[1,:], 'g-.', quadnodes.ravel(), Tlambv[2,:], 'b-o', quadnodes.ravel(), Tlambv[3,:], 'y-+') 
ax2.set(xlabel=r'# quadrature points', ylabel=r'residual error')
ax2.set_yscale('log')
ax2.set_xscale('log')
plt.show()
