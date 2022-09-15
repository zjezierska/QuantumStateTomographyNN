from cgitb import small
import qutip as qt
import numpy as np
from parameters_talitha import *


def make_data(size, h, small_d, big_d, op):
    target1s = [[] for x in range(size)]
    inputs1 = [[] for m in range(size)]
    np.random.seed(3829834)

    for i in range(size):
        t = qt.rand_dm_ginibre(N=small_d)  # generating random state

        # turning the state into 2d^2 vector - Talitha way
        full_array = np.full([big_d, big_d], 0. + 0.j)
        t_new = full_array[0:small_d, 0:small_d] = t.full()
        beginning_state = qt.Qobj(full_array)
        target1s[i] = np.concatenate((t_new.real.flatten(), t_new.imag.flatten()))  # one of expected results

        # calculating the moments of the evolution at times in tlist
        options = qt.Options()
        evolution = qt.mesolve(h, beginning_state, tlist, c_ops, [op, op ** 2], options=options)
        first_trajectory = [s - qt.expect(op, beginning_state) for s in evolution.expect[0]]  # offset from <x(0)>
        second_trajectory = (evolution.expect[1] - evolution.expect[0] ** 2).flatten()  # trajectory of <x^2> - <x>^2
        if h == H_quartic:
            np.save(f"data/{small_d}d/trajectories/H_quartic/x{i}.npy", first_trajectory)
            np.save(f"data/{small_d}d/trajectories/H_quartic/variance{i}.npy", second_trajectory)
        elif h == H_harmonic:
            np.save(f"data/{small_d}d/trajectories/H_harmonic/x{i}.npy", first_trajectory)
            np.save(f"data/{small_d}d/trajectories/H_harmonic/variance{i}.npy", second_trajectory)
        print(f"Saving {small_d}d {i}")
    
    targets = np.array(target1s)
    np.save(f'data/{small_d}d/states.npy', targets)
    print(f"Saving {small_d}d states")
