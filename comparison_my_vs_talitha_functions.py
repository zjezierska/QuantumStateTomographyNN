import tensorflow as tf
import numpy as np
import qutip as qt

d = dimp = 4


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


import keras.backend as K


def my_mean_squared_error_loss(y_true, y_pred):  # TALTHA'S CODE

    # calc rho in vector form from y_pred which is the T matrix
    input_shape = K.shape(y_pred)
    trace = K.sum(K.square(y_pred), axis=-1)
    trace = K.repeat_elements(K.expand_dims(trace, axis=-1), 2 * d ** 2, axis=-1)

    matrix_form = K.reshape(y_pred, (input_shape[0], 2, d,
                                     d))  # THIS transforms the 2*n**2 output values to two nxn matrices (real and
    # complex parts)
    transpose_matrix = K.permute_dimensions(matrix_form,
                                            (0, 1, 3, 2))  # THIS returns a tensor containing the transposed matrices

    real_elements = tf.matmul(transpose_matrix,
                              matrix_form)  # THIS performs the matrix multiplication on the matrix defined by the
    # last two indices
    rho_real = K.expand_dims(K.sum(real_elements, axis=1), axis=1)  # ,keepdims=True)

    rolled_matrix = K.tile(matrix_form, [1, 2, 1, 1])
    rolled_matrix = rolled_matrix[:, 1:-1, :, :]  # the 1 is actually length (of the relevant dimension)-1
    # print(f"rolled matrix: {rolled_matrix}")
    # print(f"transpose matrix: {transpose_matrix}")
    imag_elements = tf.matmul(transpose_matrix, rolled_matrix)
    # print(f"imag_elements: {imag_elements}")
    oneone = [[1] * dimp for _ in range(dimp)]
    twotwo = [[-1] * dimp for _ in range(dimp)]
    signed_imag = imag_elements * [oneone, twotwo]  # THIS multiplication works element-wise
    rho_imag = K.expand_dims(K.sum(signed_imag, axis=1), axis=1)

    rho_pred = K.concatenate([rho_real, rho_imag], axis=1)

    rho_pred = K.batch_flatten(rho_pred) / trace

    return K.mean(K.square(rho_pred - y_true), axis=-1)


def my_fidelity_metric(y_true, y_pred):  # for vectors ONLY - TALITHA'S CODE
    y_true = y_true[0]
    y_pred = y_pred[0]
    y_pred = y_pred[0]
    y_true = y_true[0]
    sq = y_pred ** 2
    trace = np.sum(sq)
    together = y_pred[0:dimp ** 2] + 1.j * y_pred[dimp ** 2:]
    m_pred = np.reshape(together, [dimp, dimp])
    m_pred = np.matmul(np.transpose(np.conjugate(m_pred)), m_pred) / trace
    m_true = np.reshape(y_true[0:dimp ** 2] + 1.j * y_true[dimp ** 2:], [dimp, dimp])
    rho_true = qt.Qobj(m_true)
    rho_pred = qt.Qobj(m_pred)
    return qt.fidelity(rho_true, rho_pred)


def give_back_matrix(vectr, dimen):  # turn the 2d**2 vector back into Qobj matrix - MY CODE
    vec = vectr.reshape(2, dimen ** 2)
    res = vec[:1, :] + 1j * vec[1:2, :]
    matrix = res.reshape(dimen, dimen)
    return qt.Qobj(matrix)

# x = [[1.30174119e-01, 1.65920266e-02, -8.66686400e-02, -1.18059496e-01,
#       1.65920266e-02, 1.82581525e-01, 1.27540038e-02, -1.01759478e-02,
#       -8.66686400e-02, 1.27540038e-02, 3.84429047e-01, 6.06253968e-02,
#       -1.18059496e-01, -1.01759478e-02, 6.06253968e-02, 3.02815309e-01,
#       3.73971302e-19, -7.02846096e-03, 3.12745575e-03, -1.30309130e-01,
#       7.02846096e-03, 2.43818962e-18, 4.52774413e-02, 2.08995456e-02,
#       -3.12745575e-03, -4.52774413e-02, 2.78784989e-18, 1.11893874e-01,
#       1.30309130e-01, -2.08995456e-02, -1.11893874e-01, -5.60001082e-18]]
x = [[1.30174119e-01, 1.65920266e-02, -8.66686400e-02, -1.18059496e-01,
      1.65920266e-02, 1.82581525e-01, 1.27540038e-02, -1.01759478e-02,
      -8.66686400e-02, 1.27540038e-02, 3.84429047e-01, 6.06253968e-02,
      -1.18059496e-01, -1.01759478e-02, 6.06253968e-02, 3.02815309e-01,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0]]
y = [[-0.24659091, -0.15787622, 0.9978396, 0.4808422,
      -0.9387077, 0.8632961, 0.99710786, 0.95902747,
      -0.11692601, -0.35866034, -0.9986546, 0.8786953,
      -0.89421034, -0.8624091, 0.995427, 0.86151856,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0]]

print(f"custom loss: {custom_loss(x, y)}")
print(f"Talitha's custom loss: {my_mean_squared_error_loss(x, y)}")
obj = give_back_matrix(np.matrix(y), d)
obj = (obj.dag() * obj) / (obj.dag() * obj).tr()
obj2 = give_back_matrix(np.matrix(x), d)
print(f"My fidelity: {qt.fidelity(obj, obj2)}")
print(f"Talitha's fidelity: {my_fidelity_metric(np.expand_dims(np.matrix(x), axis=0), np.expand_dims(np.matrix(y), axis=0))}")
print(f"loss check: {custom_loss(x, y) == my_mean_squared_error_loss(x, y)}")
print(f"fidelity check: {qt.fidelity(obj, obj2) == my_fidelity_metric(np.expand_dims(np.matrix(x), axis=0), np.expand_dims(np.matrix(y), axis=0))}")




