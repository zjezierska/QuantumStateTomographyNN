from nn_functions import *
import time
import numpy as np
from labellines import labelLines, labelLine

start_time = time.time()

points = dict((f"dimension{d}Qr", get_infidelities(d, num_of_points, H_quartic)) for d in d_array)
# points2 = dict((f"dimension{d}Hm", get_infidelities(d, num_of_points, H_harmonic)) for d in d_array)

#print(points["dimension2Qr"])
import numpy as np
tlist = np.linspace(0, 20, 400)
t_x = tlist[::400 // 10]
print(t_x)
print(points["dimension2Qr"])
print(points["dimension3Qr"])
print(points["dimension4Qr"])
fig, ax = plt.subplots()
ax.plot(t_x, points["dimension2Qr"], '-o', label='d = 2')
ax.plot(t_x, points["dimension3Qr"], '-o', label='d = 3')
ax.plot(t_x, points["dimension4Qr"], '-o', label='d = 4')
ax.set_yscale('log')
ax.xaxis.set_ticks(np.arange(0, 20, 2.5))
ax.set_xlabel(r'time $t \omega$')
ax.set_ylabel(r'infidelity $1 - F$')
labelLines(ax.get_lines(), fontsize=12, backgroundcolor='white')

print(f"--- {time.time() - start_time} seconds ---")

fig.savefig('infid_vs_time.svg', bbox_inches='tight')
plt.show()
