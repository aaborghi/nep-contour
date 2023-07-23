"""
@author: aaborghi
We adopt one of the examples from https://dl.acm.org/doi/10.1145/2427023.2427024 .
In particular we take the polynomial problem (quadratic eigenproblem) titled hospital.
"""

import numpy as np
from matplotlib import pyplot as plt
from src.nlevpfunc import contour_solver
from scipy.io import loadmat

# setting up the hospital example
test = loadmat('hospital.mat')
M = test['K']
C = test['C']
K = test['M']
n = 24

R = 20 #radius of the circle
mu = -0.4 #center of the circle
V = np.identity(n)

# functions for the change of variables in the integral
def phi(x):
    return mu + R * np.exp(1j * x)
def dphi(x):
    return 1j * R * np.exp((x * 1j))

# hospital quadratic eigenproblem
def T(z): 
    return (z**2)*M + z*C + K


sol = contour_solver(n, T)


q = 12 # expected number of eigenvalues inside the contour
quadNodes = 128 # number of quadrature nodes
h, circle, _ = sol.hankel(quadNodes, 40, q, 2, phi, dphi, V)

hl1, circle, _ = sol.single_loewner(quadNodes, 40, q, 2, phi, dphi, 2*R+mu)

nint = q
muq = 2 * R * np.exp(1j * np.linspace(-1, 1, num=nint)) + mu 
lambdaq = - 2 * R * np.exp(1j * np.linspace(-1, 1, num=nint)) + mu 
lvec = np.random.rand(n, nint) + np.random.rand(n, nint) * 1j
rvec = np.random.rand(n, nint) + np.random.rand(n, nint) * 1j

hl, circle, _ = sol.multi_loewner(quadNodes, q, phi, dphi, muq, lambdaq, rvec, lvec)

hl2, circle, _ = sol.multi1side_loewner(quadNodes, q, phi, dphi, muq, rvec, lvec)

fig = plt.figure()
plt.plot(h.real, h.imag, 'co', hl1.real, hl1.imag, 'b^', hl.real, hl.imag, 'g*', hl2.real, hl2.imag, 'yx', circle.real, circle.imag, 'k.')
ax = plt.gca()
ax.set_xlim(-R+mu-(R/3), 3*R+mu)
ax.set_ylim(-3*R-mu, 3*R)
plt.legend([r'Hankel', r'1Loew', r'multiLoew', r'singlemultiLoew'], loc='upper right')
plt.xlabel(r'Re($z$)')
plt.ylabel(r'Im($z$)')
fig.set_size_inches(6, 6, forward=True)
plt.show()
plt.tight_layout()
