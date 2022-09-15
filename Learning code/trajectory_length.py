from nn_functions import *
import time

infidelities = []
x_plot = []
for t in range(1, N, int(N/num_of_points)):
    x_plot.append(tlist[t])
    tlis = tlist[:t]
    init_net(tlis)
    valid_set, valid_states = train_net(tlis)
    model = tensorflow.keras.models.load_model('best_model.h5')
    validation_predict = model.predict_on_batch(valid_set)
    infidel = 0
    for i in range(n):
        val_true = give_back_matrix(valid_states[i, :])
        val_predict = give_back_matrix(validation_predict[i, :])
        infidel += 1 - qt.fidelity(val_predict, val_true)

    infidelities.append(np.mean(infidel))

# # t_x = tlist[::int(N/num_of_points)]
# plt.plot(x_plot, infidelities, '-o')
# plt.title('Infidelity based on traj. len')
# plt.ylabel(r'infidelity $1 - F$')
# plt.xlabel(r'trajectory length $t \omega$')
# plt.yscale('log')
# # plt.legend(['training set', 'validation set'], loc='upper left')
#
# print(f"--- {time.time() - start_time} seconds ---")  # how much time the code took to run
# plt.show()
