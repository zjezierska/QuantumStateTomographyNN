from newapproach_data_generator import *
import qutip as qt
import matplotlib.pyplot as plt

N = 1
d = 2
data_in, data_out = fancy_data_gen(N, H_harmonic, d, 40)


def give_back_matrix(vectr):  # turn the 2d**2 vector back into Qobj matrix
    global d
    vec = vectr.reshape(2, d ** 2)
    res = vec[:1, :] + 1j * vec[1:2, :]
    matrix = res.reshape(d, d)
    return qt.Qobj(matrix)


samplez = []
for i in range(N):
    state1 = data_out[i, :]
    state1 = give_back_matrix(state1)
    full_array = np.full([D, D], 0. + 0.j)
    full_array[0:d, 0:d] = state1.full()
    state = qt.Qobj(full_array)
    result = qt.mesolve(H_harmonic, state, tlist, c_ops=c_ops)
    bin_heights = []
    for j in range(traj_length):
        rho = result.states[j]
        rho_matrix = rho.full()
        before_sum = tf.math.multiply(rho_matrix[:, :, tf.newaxis], psi_products)
        first_sum = tf.math.reduce_sum(tf.math.real(before_sum), axis=0)
        prob_list = tf.math.reduce_sum(first_sum, axis=0)
        sample = draw_from_prob(x_list, prob_list, nof_samples_distr)
        n, bins, patches = plt.hist(sample, num_bin, density=True)
        print(f"CALCULATING {j+1}. BINS FOR {i+1}. CHECK")
        bin_heights.append(list(n))

    samplez.append([element for sublist in bin_heights for element in sublist])
samplez = np.reshape(samplez, (N, traj_length * num_bin))

fig, ax = plt.subplots(1, 2)
ax[0].plot(data_in[0, :num_bin])
ax[1].plot(samplez[0, :num_bin])
plt.show()
