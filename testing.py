from typing import final
import tensorflow as tf
import numpy as np

d = 4
x = tf.convert_to_tensor(np.random.uniform(-2,10, size=(2,2*d**2)))
y = tf.convert_to_tensor(np.random.uniform(-2,10, size=(2,2*d**2)))

def custom_loss(y_true, y_pred):
    input_shape = tf.shape(y_pred)

    trace = tf.reduce_sum(tf.square(y_pred), axis = -1)  # trace shape [batchsize, 1]

    trace = tf.tile(tf.reshape(trace,[input_shape[0], 1, 1]),multiples=[1,d,d])
    matrix_form = tf.reshape(y_pred,(input_shape[0],2,d,d))
    matrix_com = tf.complex(matrix_form[:,0,:,:], matrix_form[:,1,:,:])
    transpose_matrix = tf.transpose(matrix_com, perm=[0,2,1], conjugate=True)
    result = tf.keras.backend.batch_dot(transpose_matrix, matrix_com)
    final_shit = tf.divide(result, tf.cast(trace, tf.complex128))
    print(tf.reshape(tf.math.real(final_shit), (input_shape[0], -1)))

    final_final_shit = tf.concat([tf.reshape(tf.math.real(final_shit), (input_shape[0], -1)), tf.reshape(tf.math.imag(final_shit), (input_shape[0], -1))], axis=-1)
    print(final_final_shit)


    return tf.math.reduce_mean(tf.square(final_final_shit - y_true), axis=-1)


print(custom_loss(x, y))