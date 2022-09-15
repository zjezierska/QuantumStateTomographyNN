import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from parameters_talitha import *

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")


def custom_loss(y_true, y_pred):  # my custom loss function - maybe faster than talitha's because of tf
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


def init_net(dimm, tlis):  # creating and compiling the network
    global net, batchsize
    net = keras.models.Sequential()
    net.add(keras.layers.Dense(800, input_shape=(2 * len(tlis),), activation='sigmoid'))
    net.add(keras.layers.Dense(800, activation='sigmoid'))
    net.add(keras.layers.Dense(400, activation='sigmoid'))
    net.add(keras.layers.Dense(200, activation='sigmoid'))
    net.add(keras.layers.Dense(2 * dimm ** 2, activation='tanh'))
    net.compile(loss=custom_loss, optimizer='adam')


def train_net(y_in, y_out, valid_x, valid_y):  # training the network with some callbacks
    global epochz, patienc, n, batchsize
    callbackz = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1, mode='min'),
                 keras.callbacks.ModelCheckpoint(filepath=f'Models/best_model{y_in.shape}.h5', monitor='val_loss',
                                                 save_best_only=True, mode='min')]  # early stopping and saving the best
    # model, to use in validation set

    history = net.fit(y_in, y_out, batch_size=batchsize, epochs=epochz, validation_data=(valid_x, valid_y),
                      callbacks=callbackz)
    # training the network on (n) samples in (batchsize) batches for some epochs


def give_back_matrix(vectr, dimen):  # turn the 2d**2 vector back into Qobj matrix
    vec = vectr.reshape(2, dimen ** 2)
    res = vec[:1, :] + 1j * vec[1:2, :]
    matrix = res.reshape(dimen, dimen)
    return qt.Qobj(matrix)


def my_fidelity(vec1, vec2, d):  # normalizing vec2 and calculating fidelity between two states in vector form
    vec1 = give_back_matrix(vec1, d)
    vec2 = give_back_matrix(vec2, d)
    if vec1.isherm:
        vec2 = (vec2.dag() * vec2) / (vec2.dag() * vec2).tr()
        return qt.fidelity(vec1, vec2)
    else:
        raise ValueError('X is not Hermitian!')


dimp = 0  # the global dimp parameter needs to be used in the custom loss function - maybe in can be implemented more
# nicely


def get_infidelities(dim, num_of_pointz, h):
    global dimp
    avg_infidelities = []
    print(f"---------------BEGINNING DIMENSION {dim}---------------")
    dimp = dim

    x_traj = [np.load(f"data/{dim}d/trajectories/H_quartic/x{i}.npy") for i in range(2 * n)]
    var_traj = [np.load(f"data/{dim}d/trajectories/H_quartic/variance{i}.npy") for i in range(2 * n)]  # load the trajectories

    x_traj_valid = x_traj[n:]
    var_traj_valid = var_traj[n:]
    x_traj = x_traj[:n]
    var_traj = var_traj[:n]

    x_traj = np.array(x_traj)
    var_traj = np.array(var_traj)

    x_traj_valid = np.array(x_traj_valid)
    var_traj_valid = np.array(var_traj_valid)  # cutting the data into training and validation parts

    y_out = np.load(f'data/{dim}d/states.npy')  # load the states
    y_valid_out = y_out[n:, :]
    y_out = y_out[:n, :]
    dt = N // num_of_pointz

    for t in range(0, N, dt):  # code for each of the points
        print(f"BEGINNING {t} - DIM {dim}")

        point_traj = np.concatenate((x_traj[:, :t+dt], var_traj[:, :t+dt]), axis=1)
        point_traj_valid = np.concatenate((x_traj_valid[:, :t+dt], var_traj_valid[:, :t+dt]), axis=1)  # shortened
        # trajectories

        init_net(dim, tlist[:t+dt])
        train_net(point_traj, y_out, point_traj_valid, y_valid_out)  # initiating and training the network

        model1 = tf.keras.models.load_model(f'Models/best_model{point_traj.shape}.h5',
                                            custom_objects={'custom_loss': custom_loss})  # loading the best model

        validation_predict = model1.predict_on_batch(point_traj_valid)
        fidelities = [my_fidelity(y_valid_out[i, :], validation_predict[i, :], dim) for i in range(n)]
        avg_infidelities.append(1 - np.average(fidelities))  # average infidelity on valid_data for the point

    return avg_infidelities
