import numpy as np


x_traj = [np.load(f"data/4d/trajectories/H_quartic/x{i}.npy") for i in range(20)]
var_traj = [np.load(f"data/4d/trajectories/H_quartic/variance{i}.npy") for i in range(20)]

x = [np.load(f"quarticTrajectories_4d/q_trajectory_x{i}.npy") for i in range(20)]
var = [np.load(f"quarticTrajectories_4d/q_trajectory_var{i}.npy") for i in range(20)]

for i in range(20):
    print(f"my stuff: {x_traj[0][i]}")
    print(f"not my stuff: {x[0][i]}")
    print("-------")
