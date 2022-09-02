import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from parameters import *

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")


def custom_loss(y_true, y_pred):
    input_shape = tf.shape(y_pred)

    trace = tf.reduce_sum(tf.square(y_pred), axis=-1)  # trace shape [batchsize, 1]

    trace = tf.tile(tf.reshape(trace, [input_shape[0], 1, 1]), multiples=[1, dimp, dimp])
    matrix_form = tf.reshape(y_pred, (input_shape[0], 2, dimp, dimp))
    matrix_com = tf.complex(matrix_form[:, 0, :, :], matrix_form[:, 1, :, :])
    transpose_matrix = tf.transpose(matrix_com, perm=[0, 2, 1], conjugate=True)
    result = tf.keras.backend.batch_dot(transpose_matrix, matrix_com)
    final_stuff = tf.divide(result, tf.cast(trace, tf.complex64))

    finalfinal_stuff = tf.concat([tf.reshape(tf.math.real(final_stuff), (input_shape[0], -1)),
                                  tf.reshape(tf.math.imag(final_stuff), (input_shape[0], -1))], axis=-1)

    return tf.math.reduce_mean(tf.square(finalfinal_stuff - y_true), axis=-1)


def init_net(dimm, tlis):
    global net, batchsize
    net = keras.models.Sequential()
    net.add(keras.layers.Dense(800, input_shape=(2 * len(tlis),), activation='sigmoid'))
    net.add(keras.layers.Dense(800, activation='sigmoid'))
    net.add(keras.layers.Dense(400, activation='sigmoid'))
    net.add(keras.layers.Dense(200, activation='sigmoid'))
    net.add(keras.layers.Dense(2 * dimm ** 2, activation='tanh'))
    net.compile(loss=custom_loss, optimizer='adam')  # todo: use custom_loss


def train_net(y_in, y_out, valid_x, valid_y):
    global epochz, patienc, n, batchsize
    callbackz = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1),
                 keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                                                 save_best_only=True)]  # early stopping and saving the best
    # model, to use in validation set

    history = net.fit(y_in, y_out, batch_size=batchsize, epochs=epochz, validation_data=(valid_x, valid_y),
                      callbacks=callbackz)
    # training the network on (n) samples in (batchsize) batches for some epochs


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
    target = np.load(f'data/{dim}d/states/normal.npy')
    valid_states = np.load(f"data/{dim}d/states/valid.npy")

    x_traj = [[] for m in range(n)]
    var_traj = [[] for m in range(n)]
    x_traj_valid = [[] for m in range(n)]
    var_traj_valid = [[] for m in range(n)]
    for i in range(n):
        x_traj[i] = np.load(f"data/{dim}d/trajectories/H_quartic/normal/x{i}.npy")
        var_traj[i] = np.load(f"data/{dim}d/trajectories/H_quartic/normal/variance{i}.npy")

        x_traj_valid[i] = np.load(f"data/{dim}d/trajectories/H_quartic/validation/x{i}.npy")
        var_traj_valid[i] = np.load(f"data/{dim}d/trajectories/H_quartic/validation/variance{i}.npy")

    x_traj = np.array(x_traj)
    var_traj = np.array(var_traj)
    x_traj_valid = np.array(x_traj_valid)
    var_traj_valid = np.array(var_traj_valid)

    for t in range(0, N, N // num_of_points):
        if t == 0:
            continue

        point_traj = np.concatenate((x_traj[:, :t], var_traj[:, :t]), axis=1)
        point_traj_valid = np.concatenate((x_traj_valid[:, :t], var_traj_valid[:, :t]), axis=1)

        init_net(dim, tlist[:t])

        train_net(point_traj, target, point_traj_valid, valid_states)
        model = tf.keras.models.load_model('best_model.h5', custom_objects={'custom_loss': custom_loss})
        validation_predict = model.predict_on_batch(point_traj_valid)
        infidel = 0
        for i in range(n):
            val_true = give_back_matrix(valid_states[i, :], dim)
            val_predict1 = give_back_matrix(validation_predict[i, :], dim)
            val_predict = (val_predict1.dag() * val_predict1) / (val_predict1.dag() * val_predict1).tr()
            infidel += 1 - qt.fidelity(val_predict, val_true)

        infidelities.append(infidel / n)

    return infidelities
