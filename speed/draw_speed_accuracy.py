import tensorflow
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
import time

# <editor-fold desc="NUMBER PARAMETERS">
d = 4  # initial state dims limit
D = 40  # possible evolution dims
w = 1.0  # frequency
alpha = 5  # inverse quarticity
gamma = 0  # decoherence rate
batchsize = 512
epochz = 10000
patienc = 500
np.random.seed(3829834)
# </editor-fold>

# <editor-fold desc="QUANTUM DEFINITIONS">
a = qt.destroy(D)  # annihilation operator
x = a.dag() + a
p = 1j * (a.dag() - a)
H_quartic = p*p/4 + (x/alpha)*(x/alpha)*(x/alpha)*(x/alpha)
c_ops = [np.sqrt(gamma) * x]  # decoherence
dt = 0.01


# </editor-fold>


def make_batch(size, t_list):
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
        evolution = qt.mesolve(H_quartic, beginning_state, t_list, c_ops, [x, x ** 2])
        [j - qt.expect(x, beginning_state) for j in evolution.expect[0]]  # offset from <x(0)>
        first_trajectory = evolution.expect[0].flatten()  # trajectory of <x> - <x(0)>
        second_trajectory = (evolution.expect[1] - evolution.expect[0] ** 2).flatten()  # trajectory of <x^2> - <x>^2
        inputs1[i] = np.concatenate((first_trajectory, second_trajectory))

    targets = np.array(target1s)
    inputs = np.array(inputs1)

    return inputs, targets


def init_net(N):
    global net, d
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




val_losses = []
times = []
x_ax = np.linspace(100, 2100, num=20)
for iterator in range(100, 2100, 100):
    tlist = np.linspace(0, iterator * dt, iterator)
    inputz, targetz = make_batch(10000, tlist)
    init_net(iterator)
    callbackz = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patienc, verbose=1, mode='min')]
    beginning = time.time()
    history = net.fit(inputz, targetz, batch_size=batchsize, epochs=epochz, validation_split=0.2, callbacks=callbackz)
    val_losses.append(np.mean(history.history['val_loss']))
    times.append(time.time() - beginning)

print(f"x_ax={x_ax}")
print(f"val_losses={val_losses}")
print(f"times={times}")
# create figure and axis objects with subplots()
fig, ax = plt.subplots()
# make a plot
ax.plot(x_ax, val_losses, "-o", label='Avg validation loss')
# set x-axis label
ax.set_xlabel("Input size", fontsize=14)
# set y-axis label
ax.set_ylabel("loss function")
ax.set_yscale('log')

# twin object for two different y-axis on the sample plot
ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(x_ax, times, "--o", label='running time', color='orange')
ax2.set_ylabel("running time [s]")
fig.legend()
plt.show()
# save the plot as a file
fig.savefig('speed2.svg',
            dpi=300,
            bbox_inches='tight')
