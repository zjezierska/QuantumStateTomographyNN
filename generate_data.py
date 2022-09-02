from cgitb import small
import qutip as qt
import numpy as np
from parameters import *


def make_data(size, h, small_d, big_d, op, t_list, valid=False):
    target1s = [[] for x in range(size)]
    inputs1 = [[] for m in range(size)]
    for i in range(size):
        t = qt.rand_dm_hs(N=small_d)  # generating random state

        # turning the state into 2d^2 vector - Talitha way
        full_array = np.full([big_d, big_d], 0. + 0.j)
        t_new = full_array[0:small_d, 0:small_d] = t.full()
        beginning_state = qt.Qobj(full_array)
        target1s[i] = np.concatenate((t_new.real.flatten(), t_new.imag.flatten()))  # one of expected results

        # calculating the moments of the evolution at times in tlist
        evolution = qt.mesolve(h, beginning_state, t_list, c_ops, [op, op ** 2])
        expect = [s - qt.expect(op, beginning_state) for s in evolution.expect[0]]  # offset from <x(0)>
        first_trajectory = expect  # trajectory of <x> - <x(0)>
        second_trajectory = (evolution.expect[1] - evolution.expect[0] ** 2).flatten()  # trajectory of <x^2> - <x>^2
        #inputs1[i] = np.concatenate((first_trajectory, second_trajectory))
        if valid:
            if h == H_quartic:
                np.save(f"data/{small_d}d/trajectories/H_quartic/validation/x{i}.npy", first_trajectory)
                np.save(f"data/{small_d}d/trajectories/H_quartic/validation/variance{i}.npy", second_trajectory)
            elif h == H_harmonic:
                np.save(f"data/{small_d}d/trajectories/H_harmonic/validation/x{i}.npy", first_trajectory)
                np.save(f"data/{small_d}d/trajectories/H_harmonic/validation/variance{i}.npy", second_trajectory)
        elif h == H_quartic:
            np.save(f"data/{small_d}d/trajectories/H_quartic/normal/x{i}.npy", first_trajectory)
            np.save(f"data/{small_d}d/trajectories/H_quartic/normal/variance{i}.npy", second_trajectory)
        elif h == H_harmonic:
            np.save(f"data/{small_d}d/trajectories/H_harmonic/normal/x{i}.npy", first_trajectory)
            np.save(f"data/{small_d}d/trajectories/H_harmonic/normal/variance{i}.npy", second_trajectory)
    
    targets = np.array(target1s)
    if valid:
        np.save(f'data/{small_d}d/states/valid.npy', targets)
    else:
        np.save(f'data/{small_d}d/states/normal.npy', targets)


make_data(n, H_quartic, 2, D, x, tlist)
make_data(n, H_quartic, 2, D, x, tlist, valid=True)
make_data(n, H_quartic, 3, D, x, tlist)
make_data(n, H_quartic, 3, D, x, tlist, valid=True)
#make_data(n, H_quartic, 4, D, x, tlist)
#make_data(n, H_quartic, 4, D, x, tlist, valid=True)
