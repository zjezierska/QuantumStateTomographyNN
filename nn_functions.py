import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import tensorflow
from tensorflow import keras
from parameters import *

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


def make_data(size, h, small_d, big_d, op, t_list):  # TODO: very unoptimised(?)
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
        [s - qt.expect(op, beginning_state) for s in evolution.expect[0]]  # offset from <x(0)>
        first_trajectory = evolution.expect[0].flatten()  # trajectory of <x> - <x(0)>
        second_trajectory = (evolution.expect[1] - evolution.expect[0] ** 2).flatten()  # trajectory of <x^2> - <x>^2
        inputs1[i] = np.concatenate((first_trajectory, second_trajectory))

    targets = np.array(target1s)
    inputs = np.array(inputs1)

    return inputs, targets


def custom_loss(y_true, y_pred):
    global dimp
    sums = 0
    for i in range(batchsize):
        y_t = give_back_matrix(y_true[i].numpy(), dimp)
        y_p = give_back_matrix(y_pred[i].numpy(), dimp)
        if y_t.isherm:
            if y_p.dag() * y_p == 0:
                print(y_p)
                exit()
            y_pp = (y_p.dag() * y_p) / (y_p.dag() * y_p).tr()
            y_pp_mat = y_pp.full()
            y_pp_vec = np.concatenate((y_pp_mat.real.flatten(), y_pp_mat.imag.flatten()))
            for j in range(2 * dimp ** 2):
                sums += (y_true[i][j] - y_pp_vec[j]) ** 2
        else:
            print(y_t)
            exit()
    return sums / batchsize


def init_net(dimm, tlis):
    global net, batchsize
    net = keras.models.Sequential()
    net.add(keras.layers.Dense(800, input_shape=(2 * len(tlis),), activation='sigmoid'))
    net.add(keras.layers.Dense(800, activation='sigmoid'))
    net.add(keras.layers.Dense(400, activation='sigmoid'))
    net.add(keras.layers.Dense(200, activation='sigmoid'))
    net.add(keras.layers.Dense(2 * dimm ** 2, activation='tanh'))
    net.compile(loss=custom_loss, optimizer='adam', run_eagerly=True)


def train_net(dim, tlistt, ham):
    global epochz, patienc, n, batchsize
    callbackz = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1),
                 keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                                                 save_best_only=True)]  # early stopping and saving the best
    # model, to use in validation set

    y_in, y_out = make_data(n, ham, dim, D, x, tlistt)
    valid_x, valid_y = make_data(n, ham, dim, D, x, tlistt)
    history = net.fit(y_in, y_out, batch_size=batchsize, epochs=epochz, validation_data=(valid_x, valid_y),
                      callbacks=callbackz)
    # training the network on (n) samples in (batchsize) batches for some epochs

    return valid_x, valid_y


def give_back_matrix(vectr, dimen):  # turn the 2d**2 vector back into Qobj matrix
    vec = vectr.reshape(2, dimen ** 2)
    res = vec[:1, :] + 1j * vec[1:2, :]
    matrix = res.reshape(dimen, dimen)
    return qt.Qobj(matrix)


def draw_wigner(true_drawing, predicted_drawing):
    xvec = np.linspace(-5, 5, 200)
    W1 = qt.wigner(true_drawing, xvec, xvec)
    W2 = qt.wigner(predicted_drawing, xvec, xvec)

    wmap = qt.wigner_cmap(W1)  # can edit colormap, put it in cmap
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.contourf(xvec, xvec, W1, 100, cmap='RdBu_r')
    ax1.set_title("True state")
    ax2.contourf(xvec, xvec, W2, 100, cmap='RdBu_r')
    ax2.set_title("Predicted state")
    fig.suptitle('True vs predicted Wigner function')

    plt.show()


dimp = 0


def get_infidelities(dim, num_of_points, h):
    global dimp
    infidelities = []
    print(f"---------------BEGINNING DIMENSION {dim}---------------")
    dimp = dim
    for t in range(1, N, int(N / num_of_points)):
        tlis = tlist[:t]
        init_net(dim, tlis)
        valid_set, valid_states = train_net(dim, tlis, h)
        model = tensorflow.keras.models.load_model('best_model.h5')
        validation_predict = model.predict_on_batch(valid_set)
        infidel = 0
        for i in range(n):
            val_true = give_back_matrix(valid_states[i, :], dim)
            val_predict = give_back_matrix(validation_predict[i, :], dim)
            # val_predict = (val_predict.dag() * val_predict) / (val_predict.dag() * val_predict).tr()
            # todo: put this in custom loss function - true physical state
            infidel += 1 - qt.fidelity(val_predict, val_true)

        infidelities.append(np.abs(infidel) / n)

    return infidelities
