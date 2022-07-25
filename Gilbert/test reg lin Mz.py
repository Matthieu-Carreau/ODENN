

import numpy as np
import matplotlib.pyplot as plt

lamb = 0.3
W = 2*np.pi


def E(mu) :
    E = 64/5*lamb**2*W**2*mu**4
    E += -32/3*lamb*W*mu**3
    E += -(32/3*lamb**2*W**2-4)*mu**2
    E += 8*lamb*W*mu
    E += 4*lamb**2*W**2
    return E / 4 # 4 = length of interval

MU = np.linspace(-1, 1, 1000)
plt.plot(MU, E(MU))


def dMz(mu, t) :
    return lamb*W*(mu**2*t**2-1)



N = 500
T = np.linspace(-2, 2, N)

def E_num(mu) :
    return np.sum((dMz(mu,T)-mu)**2)/N

#plt.plot(MU, [E_num(mu) for mu in MU])
plt.show()

