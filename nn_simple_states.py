import tensorflow
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
import time

start_time = time.time()  # measuring the time of calculation

# <editor-fold desc="NUMBER PARAMETERS">
d = 4  # initial state dims limit
D = 20  # possible evolution dims
w = 1.0  # frequency
alpha = 5  # inverse quarticity
gamma = 0.01  # decoherence rate
t_lim = 20
N = 400  # number of points in a trajectory
n = 20000  # how many samples to test and validate on - see validation_split
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


def make_batch(size):  # TODO: very unoptimised(?) - takes a big % of time running
    global h
    target1s = [[] for o in range(size)]
    inputs1 = [[] for m in range(size)]
    for i in range(size):
        t = qt.rand_dm_hs(N=d)  # generating random state

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
    net = tensorflow.keras.models.Sequential()
    net.add(tensorflow.keras.layers.Dense(800, input_shape=(2 * N,), activation='sigmoid'))
    net.add(tensorflow.keras.layers.Dense(800, activation='sigmoid'))
    net.add(tensorflow.keras.layers.Dense(400, activation='sigmoid'))
    net.add(tensorflow.keras.layers.Dense(200, activation='sigmoid'))
    net.add(tensorflow.keras.layers.Dense(2 * d ** 2, activation='tanh'))
    net.compile(loss='mse', optimizer='adam')


def give_back_matrix(vectr):  # turn the 2d**2 vector back into Qobj matrix
    global d
    vec = vectr.reshape(2, d ** 2)
    res = vec[:1, :] + 1j * vec[1:2, :]
    matrix = res.reshape(d, d)
    return qt.Qobj(matrix)


def custom_loss(y_true, y_pred):  # custom loss function - not used rn, very slow
    sumd = 0
    for i in range(len(y_true)):
        sumd += (y_true[i] - y_pred[i]) ** 2
    return sumd / len(y_true)


init_net()

callbackz = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1),
             tensorflow.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                                                        save_best_only=True)]  # early stopping and saving the best
# model, maybe to use later

# PLOTTING AVG COST FUNTION IN EPOCHS
y_in, y_out = make_batch(n)
history = net.fit(y_in, y_out, batch_size=batchsize, epochs=epochz, validation_split=0.5, callbacks=callbackz)
# training the network on (n) samples in (batchsize) batches for some epochs

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss function')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper left')

# TESTING ON SOME OTHER TESTING SET
F = 0
fidel_max = 0
for i in range(batchsize):
    x_val, y_val = make_batch(1)
    test_result = net.predict(x_val, verbose=0)
    true_y = give_back_matrix(y_val)
    pred_res1 = give_back_matrix(test_result)
    pred_res = pred_res1.dag()*pred_res1 / (pred_res1.dag()*pred_res1).tr()
    fidel = qt.fidelity(pred_res, true_y)
    if fidel > fidel_max:  # I want to draw the best fit in the validation set
        fidel_max = fidel
        true_drawing = true_y
        predicted_drawing = pred_res
    F += fidel

avg_fid = F / batchsize
print(f"AVERAGE FIDELITY IN VALID. SET: {avg_fid}")

# DRAWING THE BEST FIT IN VALID. SET - looks cool
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

print(f"--- {time.time() - start_time} seconds ---")  # how much time the code took to run

plt.show()
