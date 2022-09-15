import numpy as np
import qutip as qt
# <editor-fold desc="NUMBER PARAMETERS">
D = 40  # possible evolution dims
alpha = 5  # inverse quarticity
gamma = 0  # decoherence rate
t_lim = 20
N = 400  # number of points in a trajectory
n = 10000  # idk
batchsize = 512
epochz = 10000
patienc = 500
d_array = [2, 3, 4]
num_of_points = 10
# </editor-fold>

# <editor-fold desc="QUANTUM DEFINITIONS">
a = qt.destroy(D)  # annihilation operator
x = a.dag() + a
p = 1j * (a.dag() - a)
H_quartic = p*p/4 + (x/alpha)*(x/alpha)*(x/alpha)*(x/alpha)
H_harmonic = a.dag() * a
# # A list of collapse operators - later
c_ops = [np.sqrt(gamma) * x]  # decoherence
tlist = np.linspace(0, t_lim, N)  # points in trajectory
# </editor-fold>