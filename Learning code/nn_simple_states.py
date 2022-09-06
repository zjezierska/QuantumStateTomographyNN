import tensorflow as tf
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
import time

start_time = time.time()  # measuring the time of calculation

# <editor-fold desc="NUMBER PARAMETERS">
d = 4  # initial state dims limit
D = 40  # possible evolution dims
w = 1.0  # frequency
alpha = 5  # inverse quarticity
gamma = 0  # decoherence rate
t_lim = 20
N = 400  # number of points in a trajectory
n = 10000  # how many samples to test and validate on - see validation_split
batchsize = 512
epochz = 10000
patienc = 500
# </editor-fold>

# <editor-fold desc="QUANTUM DEFINITIONS">
a = qt.destroy(D)  # annihilation operator
x = (a.dag() + a) / np.sqrt(2 * w)
p = 1j * (a.dag() - a) * np.sqrt(w / 2)
h = (- (w * (a.dag() - a)) ** 2) / 8 + (((a.dag() + a) / alpha) ** 4) / (4 * w)  # quartic potential
c_ops = [np.sqrt(gamma) * x]  # decoherence
tlist = np.linspace(0, t_lim, N)  # points in trajectory


# </editor-fold>


def make_batch(size):
    global h
    target1s = [[] for o in range(size)]
    inputs1 = [[] for m in range(size)]
    for i in range(size):
        t = qt.rand_dm_hs(N=d, seed=3829834)  # generating random state

        # turning the state into 2d^2 vector - Talitha way
        full_array = np.full([D, D], 0. + 0.j)
        t_new = full_array[0:d, 0:d] = t.full()
        beginning_state = qt.Qobj(full_array)
        target1s[i] = np.concatenate((t_new.real.flatten(), t_new.imag.flatten()))  # one of expected results

        # calculating the moments of the evolution at times in tlist
        evolution = qt.mesolve(h, beginning_state, tlist, c_ops, [x, x ** 2])
        [p - qt.expect(x, beginning_state) for p in evolution.expect[0]]  # offset from <x(0)>
        first_trajectory = evolution.expect[0].flatten()  # trajectory of <x> - <x(0)>
        second_trajectory = (evolution.expect[1] - evolution.expect[0] ** 2).flatten()  # trajectory of <x^2> - <x>^2
        inputs1[i] = np.concatenate((first_trajectory, second_trajectory))

    targets = np.array(target1s)
    inputs = np.array(inputs1)

    return inputs, targets


def init_net():
    global net, batchsize, N, d
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(800, input_shape=(2 * N,), activation='sigmoid'))
    net.add(tf.keras.layers.Dense(800, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(400, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(200, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(2 * d ** 2, activation='tanh'))
    net.compile(loss=custom_loss, optimizer='adam')


def give_back_matrix(vectr):  # turn the 2d**2 vector back into Qobj matrix
    global d
    vec = vectr.reshape(2, d ** 2)
    res = vec[:1, :] + 1j * vec[1:2, :]
    matrix = res.reshape(d, d)
    return qt.Qobj(matrix)


def custom_loss(y_true, y_pred):
    global d
    input_shape = tf.shape(y_pred)

    trace = tf.reduce_sum(tf.square(y_pred), axis=-1)  # trace shape [batchsize, 1]

    trace = tf.tile(tf.reshape(trace, [input_shape[0], 1, 1]), multiples=[1, d, d])
    matrix_form = tf.reshape(y_pred, (input_shape[0], 2, d, d))
    matrix_com = tf.complex(matrix_form[:, 0, :, :], matrix_form[:, 1, :, :])
    transpose_matrix = tf.transpose(matrix_com, perm=[0, 2, 1], conjugate=True)
    result = tf.keras.backend.batch_dot(transpose_matrix, matrix_com)
    final_stuff = tf.divide(result, tf.cast(trace, tf.complex64))

    finalfinal_stuff = tf.concat([tf.reshape(tf.math.real(final_stuff), (input_shape[0], -1)),
                                  tf.reshape(tf.math.imag(final_stuff), (input_shape[0], -1))], axis=-1)

    return tf.math.reduce_mean(tf.square(finalfinal_stuff - y_true), axis=-1)


init_net()

callbackz = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1)]  # early stopping

# PLOTTING AVG COST FUNTION IN EPOCHS
y_in, y_out = make_batch(n)
y_valid, y_valid_out = make_batch(n)
history = net.fit(y_in, y_out, batch_size=batchsize, epochs=epochz, validation_data=(y_valid, y_valid_out),
                  callbacks=callbackz)
# training the network on (n) samples in (batchsize) batches for some epochs

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'][::50])
plt.title('Model loss')
plt.ylabel('loss functions')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper left')

# TESTING ON SOME OTHER TESTING SET
F = 0
infidel_min = 0.96
infidel_max = 0
infidel = []
for i in range(n):
    x_val, y_val = make_batch(1)
    test_result = net.predict(x_val, verbose=0)
    true_y = give_back_matrix(y_val)
    pred_res1 = give_back_matrix(test_result)
    pred_res = pred_res1.dag() * pred_res1 / (pred_res1.dag() * pred_res1).tr()
    infidelity = 1 - qt.fidelity(pred_res, true_y)
    if infidelity < infidel_min:
        infidel_min = infidelity
        true_drawing_lowest = true_y
        pred_drawing_lowest = pred_res
    if infidelity > infidel_max:  # I want to draw the best and worst fit in the testing set
        infidel_max = infidelity
        true_drawing_highest = true_y
        pred_drawing_highest = pred_res


# DRAWING THE BEST FIT IN VALID. SET - looks cool
xvec = np.linspace(-5, 5, 200)
W1_good = qt.wigner(true_drawing_lowest, xvec, xvec)
W2_good = qt.wigner(pred_drawing_lowest, xvec, xvec)
W1_bad = qt.wigner(true_drawing_highest, xvec, xvec)
W2_bad = qt.wigner(pred_drawing_highest, xvec, xvec)

wmap = qt.wigner_cmap(W1_good)  # can edit colormap, put it in cmap
fig, axs = plt.subplots(2, 2)
axs[0, 0].contourf(xvec, xvec, W1_good, 100, cmap='RdBu_r')
axs[0, 0].set_title("True state - best infidelity")

axs[1, 0].contourf(xvec, xvec, W2_good, 100, cmap='RdBu_r')
axs[1, 0].set_title("Predicted state - best infidelity")

axs[0, 1].contourf(xvec, xvec, W1_bad, 100, cmap='RdBu_r')
axs[0, 1].set_title("True state - worst infidelity")

axs[1, 1].contourf(xvec, xvec, W2_bad, 100, cmap='RdBu_r')
axs[1, 1].set_title("Predicted state - worst infidelity")

fig.suptitle('True vs predicted Wigner function')
plt.savefig('fancywignerfunctions.svg', bbox_inches='tight')

print(f"--- {time.time() - start_time} seconds ---")  # how much time the code took to run

plt.show()
