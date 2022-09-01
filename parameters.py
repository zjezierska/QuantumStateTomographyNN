import numpy as np
import qutip as qt
# <editor-fold desc="NUMBER PARAMETERS">
D = 20  # possible evolution dims
w = 1.0  # frequency
alpha = 5  # inverse quarticity
gamma = 0  # decoherence rate
t_lim = 20
N = 400  # number of points in a trajectory
n = 10000  # idk
batchsize = 512
epochz = 10000
patienc = 500
d_array = [2]
np.random.seed(3829834)
num_of_points = 3
# </editor-fold>

# <editor-fold desc="QUANTUM DEFINITIONS">
a = qt.destroy(D)  # annihilation operator
x = (a.dag() + a) / np.sqrt(2 * w)
p = 1j * (a.dag() - a) * np.sqrt(w / 2)
H_quartic = (- (w * (a.dag() - a)) ** 2) / 8 + (((a.dag() + a) / alpha) ** 4) / (4 * w)
H_harmonic = w * (a.dag() * a + 1 / 2)
# # A list of collapse operators - later
c_ops = [np.sqrt(gamma) * x]  # decoherence
tlist = np.linspace(0, t_lim, N)  # points in trajectory
# </editor-fold>