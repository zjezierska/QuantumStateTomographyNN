from qutip import *
import numpy as np
import matplotlib.pyplot as plt

w = 1  # oscillator frequency
alpha = 5
gamma = 0.3
d = 4


a = destroy(d)  # oscillator annihilation operator
x = (a.dag() + a)/np.sqrt(2*w)
p = 1j*(a.dag() - a)*np.sqrt(w/2)

rho1 = coherent_dm(N=d, alpha=1.0)  # initial state
rho0 = rand_dm_hs(N=d)  # initial state
# H = w * (a.dag() * a + 1/2)  # Hamiltonian - oscillator
h = (- (w * (a.dag() - a)) ** 2) / 8 + (((a.dag() + a) / alpha) ** 4) / (4 * w)

# A list of collapse operators - later
c_ops = [gamma * commutator(x, commutator(x, rho0))]

tlist = np.linspace(0, 200, 400)

# request that the solver return the expectation value of the photon number state operator
result = mesolve(h, rho0, tlist, c_ops, [x, x ** 2])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('**4 potential')
ax1.plot(tlist, result.expect[0] - expect(x, rho0))
ax1.set_title(r'$\left\langle \hat{x} \right\rangle - \left\langle \hat{x}(0) \right\rangle$')
ax2.plot(tlist, result.expect[1] - result.expect[0]**2)
ax2.set_title(r'$\left< \hat{x}^2 \right> - \left< \hat{x} \right>^2$')
plt.show()
