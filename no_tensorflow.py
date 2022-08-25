import numpy as np

import matplotlib.pyplot as plt  # for plotting
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300  # highres display


def net_f_df(z):
    val = 1 / (1 + np.exp(-z))
    return val, np.exp(-z) * (val ** 2)  # only works for sigmoid


def apply_layer(y, w, b):
    z = np.dot(y, w) + b
    return net_f_df(z)


def apply_net(y_in):  # going forward
    global y_layer, df_layer
    global Weights, Biases, Nlayers

    y = y_in
    y_layer[0] = y
    for i in range(Nlayers):
        y, df = apply_layer(y, Weights[i], Biases[i])
        y_layer[i + 1] = y
        df_layer[i] = df
    return y


def backward_step(delta, w, df):  # new lower delta
    return np.dot(delta, np.transpose(w)) * df


def backprop(y_target):
    global y_layer, df_layer, Weights, Biases, Nlayers
    global dw_layer, db_layer
    global batchsize

    delta = (y_layer[-1] - y_target) * df_layer[-1]
    dw_layer[-1] = np.dot(np.transpose(y_layer[-2]), delta) / batchsize
    db_layer[-1] = delta.sum(0) / batchsize

    for j in range(Nlayers - 1):
        delta = backward_step(delta, Weights[-1 - j], df_layer[-2 - j])
        dw_layer[-2 - j] = np.dot(np.transpose(y_layer[-3 - j]), delta)/batchsize
        db_layer[-2 - j] = delta.sum(0) / batchsize


def update(eta):
    global dw_layer, db_layer, Weights, Biases

    for i in range(Nlayers):
        Weights[i] -= eta * dw_layer[i]
        Biases[i] -= eta * db_layer[i]


def train_net(y_in, y_target, eta):
    global y_out_result

    y_out_result = apply_net(y_in)
    backprop(y_target)
    update(eta)

    cost_func = ((y_target - y_out_result) ** 2).sum() / batchsize

    return cost_func


LayerSizes = [2, 100, 100, 1]
batchsize = 200
batches = 2000
eta = 0.01
Nlayers = len(LayerSizes) - 1
costs = np.zeros(batches)

# initialize random weights and biases for all layers (except input of course)
Weights = [np.random.uniform(low=-1, high=+1, size=[LayerSizes[j], LayerSizes[j + 1]]) for j in range(Nlayers)]
Biases = [np.random.uniform(low=-1, high=+1, size=LayerSizes[j + 1]) for j in range(Nlayers)]

# set up all the helper variables
y_layer = [np.zeros([batchsize, LayerSizes[j]]) for j in range(Nlayers + 1)]
df_layer = [np.zeros([batchsize, LayerSizes[j + 1]]) for j in range(Nlayers)]
dw_layer = [np.zeros([LayerSizes[j], LayerSizes[j + 1]]) for j in range(Nlayers)]
db_layer = [np.zeros(LayerSizes[j + 1]) for j in range(Nlayers)]


def myFunc(x, y):  # example function
    return x * np.exp(-x**2 - y**2)


def make_batch():
    global batchsize

    inputs = np.random.uniform(low=-0.5, high=+0.5, size=[batchsize, 2])
    targets = np.zeros([batchsize, 1])  # must have right dimensions
    targets[:, 0] = myFunc(inputs[:, 0], inputs[:, 1])
    return inputs, targets


for k in range(batches):
    y_in, y_target = make_batch()
    costs[k] = train_net(y_in, y_target, eta)

plt.plot(costs)
plt.title("Cost function during training")
plt.show()
