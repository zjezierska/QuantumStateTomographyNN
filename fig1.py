from nn_functions import *
import time
import numpy as np
from labellines import labelLines, labelLine

start_time = time.time()

points = dict((f"dimension{d}Qr", get_infidelities(d, num_of_points, H_quartic)) for d in d_array)
# points2 = dict((f"dimension{d}Hm", get_infidelities(d, num_of_points, H_harmonic)) for d in d_array)

t_x = tlist[::int(N / num_of_points)]
print(f"t_x = {t_x}")
print("points = ", points["dimension2Qr"])
plt.plot(t_x, points["dimension2Qr"], '-o', label='d = 2')
# plt.plot(t_x, points["dimension3Qr"], '-o', label='d = 3')
# plt.plot(t_x, points["dimension4Qr"], '-o', label='d = 4')
plt.yscale('log')
plt.legend()
plt.xlabel(r'time $t \omega$')
plt.ylabel(r'infidelity $1 - F$')
# labelLine(plt.gca().get_lines(), 0.6,
#           label=r"$Re=${}".format(plt.gca().get_lines().get_label()),
#           ha="left",
#           va="bottom",
#           align=False,
#           backgroundcolor="none",
#           )

print(f"--- {time.time() - start_time} seconds ---")

plt.savefig('infid_vs_time.svg', bbox_inches='tight')
plt.show()
