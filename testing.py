import matplotlib.pyplot as plt

plt.plot([4.01002506, 8.02005013, 12.03007519, 16.04010025], [0.08811091522985043, 0.061053355310527635, 0.05192228998982379, 0.03554449758644908],'-o',label="d = 2")
plt.yscale('log')
plt.legend()
plt.xlabel(r'time $t \omega$')
plt.ylabel(r'infidelity $1 - F$')
plt.savefig('infid_vs_time4d.svg', bbox_inches='tight')
plt.show()