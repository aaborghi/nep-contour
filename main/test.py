"""
@author: aaborghi
This code is to breifly test the implementation of the eigensolvers with a simple 
2x2 delay eigenproblem. 
"""
from scipy.special import lambertw
from numpy.linalg import eig
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer

from src.nlevpfunc import contour_solver

T0 = np.array([[2,0],[0,0.000001]])
T1 = np.array([[2.9,0],[0,0.5]])
n = 2

R = 5
mu = -0.5
V = np.identity(2)

def T(z):
    return z * np.identity(n) + T0 + T1 * np.exp(-z)

def lambertW(ktimes,tau,T1,T0):
    Lamb = np.array(np.zeros(n))
    for j in range(-ktimes,ktimes):
            Wk = lambertw(tau * np.negative(np.diagonal(T1)) * np.exp(tau * np.diagonal(T0)), k=j)
            Wk = np.array(Wk)
            Sk = (np.diag(Wk)/tau) - T0
            Lambk, v = eig(Sk)
            Lambk = np.array(Lambk)
            Lamb = np.concatenate((Lamb,Lambk), axis=0)
            lenLamb = len(Lamb)
    Lamb = Lamb[n:lenLamb]
    return Lamb

def phi(x):
    return mu + R * np.exp(1j * x)

def dphi(x):
    return 1j * R * np.exp((x * 1j))


sol = contour_solver(n, T)
l = lambertW(10,1,T1,T0)

q = 4

start = timer()
h, circle, eig = sol.hankel(200, 40, q, 4, phi, dphi, V)
end = timer()
print('timing for Hankel: ' + repr(end-start))

start = timer()
hl1, circle, eig = sol.single_loewner(200, 40, q, 4, phi, dphi, 2*R+mu)
end = timer()
print('timing for singleLoewner: ' + repr(end-start))


muq = 2 * R * np.exp(1j * np.linspace(-1, 1, num=q)) + mu
lambdaq = - 2 * R * np.exp(1j * np.linspace(-1, 1, num=q)) + mu
lvec = np.random.rand(n, q) + np.random.rand(n, q) * 1j
rvec = np.random.rand(n, q) + np.random.rand(n, q) * 1j

start = timer()
hl, circle, eig = sol.multi_loewner(200, q, phi, dphi, muq, lambdaq, rvec, lvec)
end = timer()
print('timing for multiLoewner: ' + repr(end-start))

start = timer()
hl2, circle, eig = sol.multi1side_loewner(200, q, phi, dphi, muq, rvec, lvec)
end = timer()
print('timing for multi1sideLoewner: ' + repr(end-start))

fig = plt.figure()
plt.plot(l.real, l.imag, 'ro')
plt.plot(h.real, h.imag, 'co', hl1.real, hl1.imag, 'b^')
plt.plot(hl.real, hl.imag, 'g*', hl2.real, hl2.imag, 'yx', circle.real, circle.imag, 'k.')
ax = plt.gca()
ax.set_xlim(-R+mu-(R/3), 3*R+mu)
ax.set_ylim(-3*R-mu, 3*R)
plt.legend([r'LambertW', r'Hankel', r'1Loew', r'multiLoew', r'singlemultiLoew'], loc='upper right')
plt.xlabel(r'Re($z$)')
plt.ylabel(r'Im($z$)')
fig.set_size_inches(5, 5, forward=True)
plt.show()
plt.tight_layout()

