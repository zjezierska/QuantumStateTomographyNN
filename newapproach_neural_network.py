from newapproach_data_generator import *
import qutip as qt
import tensorflow as tf
import matplotlib.pyplot as plt

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")

# NEW APPROACH - harmonic ham., P(x) ...


def custom_loss(y_true, y_pred):  # MY CUSTOM LOSS FUNCTION - same as in Talitha's approach
    input_shape = tf.shape(y_pred)

    trace = tf.reduce_sum(tf.square(y_pred), axis=-1)  # trace shape [batchsize, 1]

    trace = tf.tile(tf.reshape(trace, [input_shape[0], 1, 1]), multiples=[1, d, d])
    matrix_form = tf.reshape(y_pred, (input_shape[0], 2, d, d))  # turn vectors into matrices
    matrix_com = tf.complex(matrix_form[:, 0, :, :], matrix_form[:, 1, :, :])  # connect matrices into complex matrices
    transpose_matrix = tf.transpose(matrix_com, perm=[0, 2, 1], conjugate=True)  # complex conjugate of matrices
    result = tf.keras.backend.batch_dot(transpose_matrix, matrix_com)  # M.dag() * M
    final_stuff = tf.divide(result, tf.cast(trace, tf.complex64))  # previous / trace - normalisation

    finalfinal_stuff = tf.concat([tf.reshape(tf.math.real(final_stuff), (input_shape[0], -1)),
                                  tf.reshape(tf.math.imag(final_stuff), (input_shape[0], -1))], axis=-1)  # turning
    # it back into the vector

    return tf.math.reduce_mean(tf.square(finalfinal_stuff - y_true), axis=-1)  # MSE calculation


def init_net():  # creating and compiling the network
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(800, input_shape=(num_bin * traj_length,), activation='sigmoid'))
    net.add(tf.keras.layers.Dense(800, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(400, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(200, activation='sigmoid'))
    net.add(tf.keras.layers.Dense(2 * d ** 2, activation='tanh'))
    net.compile(loss=custom_loss, optimizer='adam')
    return net


def give_back_matrix(vectr):  # turn the 2d**2 vector back into Qobj matrix
    global d
    vec = vectr.reshape(2, d ** 2)
    res = vec[:1, :] + 1j * vec[1:2, :]
    matrix = res.reshape(d, d)
    return qt.Qobj(matrix)


def my_fidelity(vec1, vec2):  # normalizing vec2 and calculating fidelity between two states in vector form
    vec1 = give_back_matrix(vec1)
    vec2 = give_back_matrix(vec2)
    if vec1.isherm:
        vec2 = (vec2.dag() * vec2) / (vec2.dag() * vec2).tr()
        return qt.fidelity(vec1, vec2)
    else:
        raise ValueError('X is not Hermitian!')


samples = 2000
batchsize = 256
d = 4  # beginning dim

model = init_net()  # creating the network
data_in, data_out = fancy_data_gen(samples, H_harmonic, d, 40)
data_in_valid, data_out_valid = fancy_data_gen(samples, H_harmonic, d, 40)  # generating the data

callbackz = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1, mode='min'),
             tf.keras.callbacks.ModelCheckpoint(filepath=f'Models/best_model_draw.h5', monitor='val_loss',
                                                save_best_only=True, mode='min')]  # early stopping and saving the best
# model, to use in validation set

history = model.fit(data_in, data_out, batch_size=batchsize, epochs=epochz,
                    validation_data=(data_in_valid, data_out_valid),
                    callbacks=callbackz)  # training the network

model1 = tf.keras.models.load_model(f'Models/best_model_draw.h5',
                                    custom_objects={'custom_loss': custom_loss})  # loading the best model for the
# validation set

avg_infidelities = []
validation_predict = model1.predict(data_in_valid, batch_size=batchsize, verbose=1)  # use the best model on valid_data
fidelities = [my_fidelity(data_out_valid[i, :], validation_predict[i, :]) for i in range(samples)]
print(1 - np.average(fidelities))  # average INfidelity in validation set

# # DRAWING WIGNER STUFF
# fidelity_worst = 1
# fidelity_best = 0
# infidel = []
# previous = 0
#
# min_index = fidelities.index(min(fidelities))
# max_index = fidelities.index(max(fidelities))
# print(f"best fidelity: {max(fidelities)}")
# print(f"worst fidelity: {min(fidelities)}")
#
# mini_norm = give_back_matrix(validation_predict[min_index, :])
# pred_drawing_worst = (mini_norm.dag() * mini_norm) / (mini_norm.dag() * mini_norm).tr()
#
# maxi_norm = give_back_matrix(validation_predict[max_index, :])
# pred_drawing_best = (maxi_norm.dag() * maxi_norm) / (maxi_norm.dag() * maxi_norm).tr()
#
# true_drawing_worst = give_back_matrix(data_out_valid[min_index, :])
# true_drawing_best = give_back_matrix(data_out_valid[max_index, :])
#
# # DRAWING THE BEST AND WORST FIT IN VALID. SET - looks cool
# xvec = np.linspace(-5, 5, 200)
# W1_good = qt.wigner(true_drawing_best, xvec, xvec)
# W2_good = qt.wigner(pred_drawing_best, xvec, xvec)
# W1_bad = qt.wigner(true_drawing_worst, xvec, xvec)
# W2_bad = qt.wigner(pred_drawing_worst, xvec, xvec)
#
# wmap = qt.wigner_cmap(W1_good)  # can edit colormap, put it in cmap
# fig, axs = plt.subplots(2, 2)
# plott = axs[0, 0].contourf(xvec, xvec, W1_good, 100, cmap='RdBu_r')
# axs[0, 0].set_title("True state - best fidelity")
#
# axs[0, 1].contourf(xvec, xvec, W2_good, 100, cmap='RdBu_r')
# axs[0, 1].set_title("Predicted state - best fidelity")
#
# axs[1, 0].contourf(xvec, xvec, W1_bad, 100, cmap='RdBu_r')
# axs[1, 0].set_title("True state - worst fidelity")
#
# axs[1, 1].contourf(xvec, xvec, W2_bad, 100, cmap='RdBu_r')
# axs[1, 1].set_title("Predicted state - worst fidelity")
#
# fig.suptitle('True vs predicted Wigner function')
# fig.tight_layout()
# fig.colorbar(plott, ax=axs[:, :], location='right')
# plt.savefig('fancywignerfunctions_10k.svg', bbox_inches='tight')

x_ax = np.arange(0, len(history.history['loss']), 50)
plt.plot(history.history['loss'])
plt.plot(x_ax, history.history['val_loss'][::50])  # plot both losses during training
plt.title('Model loss')
plt.ylabel('loss functions')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['training set', 'validation set'], loc='upper left')
plt.savefig('new_loss_functions.png', bbox_inches='tight', dpi=300)
plt.show()
