from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from numpy import array, zeros, exp, random, dot, shape, transpose, reshape, meshgrid, linspace, sqrt, argmax
import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300  # highres display


def load_data():
    f = open('mnist.pkl', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    global training_inputs, training_results
    global validation_inputs, validation_results
    global test_inputs, test_results
    global num_samples, numpixels, num_test_samples

    tr_d, va_d, te_d = load_data()

    num_samples = len(tr_d[0])
    training_inputs = zeros([num_samples, numpixels])
    training_results = zeros([num_samples, 10])
    for j in range(num_samples):
        training_inputs[j, :] = reshape(tr_d[0][j], numpixels)
        training_results[j, :] = vectorized_result(tr_d[1][j])
    #    validation_inputs = [reshape(x, (numpixels)) for x in va_d[0]]
    #    validation_results = [vectorized_result(y) for y in va_d[1]]

    num_test_samples = len(te_d[0])
    test_inputs = zeros([num_test_samples, numpixels])
    test_results = zeros([num_test_samples, 10])
    for j in range(num_test_samples):
        test_inputs[j, :] = reshape(te_d[0][j], numpixels)
        test_results[j, :] = vectorized_result(te_d[1][j])


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = zeros(10)
    e[j] = 1.0
    return e


def init_net():
    global net, numpixels
    net = Sequential()
    net.add(Dense(30, input_shape=(numpixels,), activation='relu'))
    net.add(Dense(10, activation='softmax'))
    net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


def test_on(start, stop, dontprint=False):
    global test_inputs, test_results
    global net, predictions_probs, predictions, true_labels

    predictions_probs = net.predict_on_batch(test_inputs[start:stop, :])
    predictions = argmax(predictions_probs, axis=1)
    if not dontprint:
        print("Predictions: ", predictions)
        true_labels = argmax(test_results[start:stop, :], axis=1)
        print("True labels: ", true_labels)


numpixels = 784
load_data_wrapper()
init_net()

batchsize = 100
batches = int(num_samples / batchsize) - 1
costs = zeros(batches)
for j in range(batches):
    costs[j] = net.train_on_batch(training_inputs[j * batchsize:(j + 1) * batchsize, :],
                                  training_results[j * batchsize:(j + 1) * batchsize, :])[0]
plt.plot(costs)
test_on(0, 20)
plt.show()
