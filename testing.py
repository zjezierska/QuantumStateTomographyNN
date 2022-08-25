from nn_functions import *
import time
import numpy as np
from labellines import labelLines, labelLine

t_x = [0., 5.01253133, 10.02506266, 15.03759398]
points = [0.03363197109647397, 6.1373767244567335e-06, 0.0002900012692905648, 0.000102245946767001]
plt.plot(t_x, points, '-o', label='d = 2')
# plt.plot(t_x, points["dimension3Qr"], '-o', label='d = 3')
# plt.plot(t_x, points["dimension4Qr"], '-o', label='d = 4')
plt.yscale('log')
plt.legend()
plt.xlabel(r'time $t \omega$')
plt.ylabel(r'infidelity $1 - F$')
plt.savefig('infid_vs_time.svg', bbox_inches='tight')
plt.show()