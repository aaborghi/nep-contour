"""
@author: aaborghi

In this code we test the contour integral eigensolvers against the delay 
eigenvalue problem. Using a circle of radius 0.2 and centered in 0 as contour 
we expect 11 eigenvalues inside.
To compute the real solutions of the delay eigenproblem we adopt the Lambert-W function
"""
from numpy import array
from numpy import zeros
from numpy import concatenate
from numpy import negative
from numpy import exp
from numpy import diag
from numpy import diagonal
from scipy.special import lambertw
import numpy as np
from matplotlib import pyplot as plt
from src.nlevpfunc import contour_solver


n = 50 #size of the NEP
e0 = np.geomspace(-(10**10),-(10**(-4)),50)
E0 = -np.diag(e0)
R = 0.2 #radius of the circle
mu = 0 #center of the circle
V = np.identity(n)

# function for the change of variables in the integral
def phi(x):
    return mu + R * np.exp(1j * x)
def T(z):
    return z*np.eye(n) + 0.015*np.exp(-z*8)*np.eye(n) + E0


# setup of the Lambert-W function for solving the delay eigenvalue problem
def lambertW(ktimes,tau,T0,T1):
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
    return Lamb
def dphi(x):
    return 1j * R * np.exp((x * 1j))


l = lambertW(100,8,E0,0.015*np.eye(n))

sol = contour_solver(n, T)


q = 11 #number of expected eigenvalues in the contour (i.e. the circle)
quadNodes = 100 #number of quadrature nodes

h, circle, _ = sol.hankel(quadNodes, 40, q, 3, phi, dphi, V)

hl1, circle, _ = sol.single_loewner(quadNodes, 40, q, 3, phi, dphi, 2*R+mu)

nint = q
muq = 2 * R * np.exp(1j * np.linspace(-1, 1, num=nint)) + mu
lambdaq = - 2 * R * np.exp(1j * np.linspace(-1, 1, num=nint)) + mu
lvec = np.random.rand(n, nint) + np.random.rand(n, nint)*1j
rvec = np.random.rand(n, nint) + np.random.rand(n, nint)*1j
hl, circle, _ = sol.multi_loewner(quadNodes, q, phi, dphi, muq, lambdaq, rvec, lvec)
hl2, circle, _ = sol.multi1side_loewner(quadNodes, q, phi, dphi, muq, rvec, lvec)

fig, (ax1,ax2) = plt.subplots(1, 2)
ax1.plot(l.real, l.imag, 'ro')
ax1.plot(h.real, h.imag, 'co', hl1.real, hl1.imag, 'b^', hl.real, hl.imag, 'g*', hl2.real, hl2.imag, 'yx', circle.real, circle.imag, 'k.')
ax1.set_xlim([-1., 1.])
ax1.set_ylim([-5, 5])
ax2.plot(l.real, l.imag, 'ro')
ax2.plot(h.real, h.imag, 'co', hl1.real, hl1.imag, 'b^', hl.real, hl.imag, 'g*', hl2.real, hl2.imag, 'yx', circle.real, circle.imag, 'k.')
ax2.set_xlim([-.2, .2])
ax2.set_ylim([-.2, .2])
ax1.legend([r'LambertW',r'Hankel', r'1Loew', r'multiLoew', r'singlemultiLoew'], loc='upper right')
ax1.set(xlabel=r'Re($z$)', ylabel=r'Im($z$)')
ax2.set(xlabel=r'Re($z$)')
plt.show()