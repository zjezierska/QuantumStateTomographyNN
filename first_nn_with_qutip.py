import tensorflow as tf
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
import time

start_time = time.time()  # measuring the time of calculation

# <editor-fold desc="NUMBER PARAMETERS">
D = 40  # possible evolution dims
alpha = 5  # inverse quarticity
gamma = 0  # decoherence rate
t_lim = 20  # like in talitha's paper
N = 400  # number of points in a trajectory
n = 10000  # number of training samples
batchsize = 512
epochz = 10000
patienc = 500  # patience for early stopping
d = 4  # beginning state dims
# </editor-fold>

# <editor-fold desc="QUANTUM DEFINITIONS">
a = qt.destroy(D)  # annihilation operator
x = a.dag() + a
p = 1j * (a.dag() - a)
H_quartic = p * p / 4 + (x / alpha) * (x / alpha) * (x / alpha) * (x / alpha)
H_harmonic = a.dag() * a
c_ops = [np.sqrt(gamma) * x]  # decoherence - position
tlist = np.linspace(0, t_lim, N)  # points in trajectory
# </editor-fold>


def make_data(size):
    target1s = [[] for o in range(size)]
    inputs1 = [[] for m in range(size)]

    print(f"----MAKING NEW {size} DATA----")
    for i in range(size):
        t = qt.rand_dm_hs(N=d)  # generating random state

        # turning the state into 2d^2 vector - Talitha way
        full_array = np.full([D, D], 0. + 0.j)
        t_new = full_array[0:d, 0:d] = t.full()
        beginning_state = qt.Qobj(full_array)
        target1s[i] = np.concatenate((t_new.real.flatten(), t_new.imag.flatten()))  # one of expected results
        print(f"Making {i} state")

        # calculating the moments of the evolution at times in tlist
        evolution = qt.mesolve(H_quartic, beginning_state, tlist, c_ops, [x, x ** 2])
        [p - qt.expect(x, beginning_state) for p in evolution.expect[0]]  # offset from <x(0)>
        first_trajectory = evolution.expect[0].flatten()  # trajectory of <x> - <x(0)>
        second_trajectory = (evolution.expect[1] - evolution.expect[0] ** 2).flatten()  # trajectory of <x^2> - <x>^2
        inputs1[i] = np.concatenate((first_trajectory, second_trajectory))
        print(f"Making {i} trajectory")

    targets = np.array(target1s)
    inputs = np.array(inputs1)

    return inputs, targets


def init_net():  # creating and compiling the network
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


def custom_loss(y_true, y_pred):  # my custom loss function - maybe faster than talitha's because of tensorflow
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


def my_fidelity(vec1, vec2):  # normalizing vec2 and calculating fidelity between two states in vector form
    vec1 = give_back_matrix(vec1)
    vec2 = give_back_matrix(vec2)
    if vec1.isherm:
        vec2 = (vec2.dag() * vec2) / (vec2.dag() * vec2).tr()
        return qt.fidelity(vec1, vec2)
    else:
        raise ValueError('X is not Hermitian!')


x_traj = [np.load(f"data/{d}d/trajectories/H_quartic/x{i}.npy") for i in range(2 * n)]
var_traj = [np.load(f"data/{d}d/trajectories/H_quartic/variance{i}.npy") for i in range(2 * n)]  # load data

x_traj_valid = x_traj[n:]
var_traj_valid = var_traj[n:]
x_traj = x_traj[:n]
var_traj = var_traj[:n]

x_traj = np.array(x_traj)
var_traj = np.array(var_traj)
y_in = np.concatenate((x_traj, var_traj), axis=1)

x_traj_valid = np.array(x_traj_valid)
var_traj_valid = np.array(var_traj_valid)
y_valid = np.concatenate((x_traj_valid, var_traj_valid), axis=1)  # dividing data into training and valid

y_out = np.load(f'data/{d}d/states.npy')  # loading states
y_valid_out = y_out[n:, :]
y_out = y_out[:n, :]

init_net()  # initiate the network

callbackz = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=2, mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath='Models/best_model.h5', monitor='val_loss',
                                                save_best_only=True, mode='min')]

history = net.fit(y_in, y_out, batch_size=batchsize, epochs=epochz, validation_data=(y_valid, y_valid_out),
                  callbacks=callbackz)
# training the network on (n) samples in (batchsize) batches for some epochs


# plotting the loss function
# x_ax = np.arange(0, len(history.history['loss']), 50)
# plt.plot(history.history['loss'])
# plt.plot(x_ax, history.history['val_loss'][::50])
# plt.title('Model loss')
# plt.ylabel('loss functions')
# plt.xlabel('epoch')
# plt.yscale('log')
# plt.legend(['training set', 'validation set'], loc='upper left')


model1 = tf.keras.models.load_model(f'best_model.h5',
                                    custom_objects={'custom_loss': custom_loss})  # loading the best model

avg_infidelities = []
validation_predict = model1.predict(y_valid, batch_size=batchsize, verbose=1)
fidelities = [my_fidelity(y_valid_out[i, :], validation_predict[i, :], d) for i in range(n)]
print(1 - np.average(fidelities))  # average infidelity in validation set

# DRAWING WIGNER STUFF
fidelity_worst = 1
fidelity_best = 0
infidel = []
previous = 0
# can be done with y_valid, y_valid_out
for i in range(n):
    x_val, y_val = make_data(1)
    x_val1 = x_val.flatten()
    y_val1 = y_val.flatten()
    test_result = net.predict(x_val, verbose=1)

    rho_true = give_back_matrix(y_val)
    rho_pred = give_back_matrix(test_result)
    rho_pred = (rho_pred.dag() * rho_pred) / (rho_pred.dag() * rho_pred).tr()
    fidelity = qt.fidelity(rho_true, rho_pred)

    if fidelity < fidelity_worst:  # I want to draw the best and worst fit in the testing set
        fidelity_worst = fidelity
        true_drawing_worst = rho_true
        pred_drawing_worst = rho_pred

    if fidelity > fidelity_best:
        fidelity_best = fidelity
        true_drawing_best = rho_true
        pred_drawing_best = rho_pred


print(fidelity_best)
print(fidelity_worst)
# DRAWING THE BEST AND WORST FIT IN VALID. SET - looks cool
xvec = np.linspace(-5, 5, 200)
W1_good = qt.wigner(true_drawing_best, xvec, xvec)
W2_good = qt.wigner(pred_drawing_best, xvec, xvec)
W1_bad = qt.wigner(true_drawing_worst, xvec, xvec)
W2_bad = qt.wigner(pred_drawing_worst, xvec, xvec)

wmap = qt.wigner_cmap(W1_good)  # can edit colormap, put it in cmap
fig, axs = plt.subplots(2, 2)
plott = axs[0, 0].contourf(xvec, xvec, W1_good, 100, cmap='RdBu_r')
axs[0, 0].set_title("True state - best fidelity")

axs[0, 1].contourf(xvec, xvec, W2_good, 100, cmap='RdBu_r')
axs[0, 1].set_title("Predicted state - best fidelity")

axs[1, 0].contourf(xvec, xvec, W1_bad, 100, cmap='RdBu_r')
axs[1, 0].set_title("True state - worst fidelity")

axs[1, 1].contourf(xvec, xvec, W2_bad, 100, cmap='RdBu_r')
axs[1, 1].set_title("Predicted state - worst fidelity")

fig.suptitle('True vs predicted Wigner function')
fig.tight_layout()
fig.colorbar(plott, ax=axs[:, :], location='right')
plt.savefig('fancywignerfunctions_3.svg', bbox_inches='tight')

print(f"--- {time.time() - start_time} seconds ---")  # how much time the code took to run
plt.show()
