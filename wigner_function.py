from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


d = 4
D = 20
w = 1.0
alpha = 5
rho0 = rand_dm_hs(N=d)  # initial state
rho1 = coherent_dm(N=d, alpha=1.0)

full_array = np.full([D, D], 0. + 0.j)
full_array[0:d, 0:d] = rho0.full()
full_state = Qobj(full_array)

xvec = np.linspace(-5, 5, 200)
a = destroy(d)  # oscillator annihilation operator
x = (a.dag() + a) / np.sqrt(2 * w)
p = 1j * (a.dag() - a) * np.sqrt(w / 2)
W_coherent = wigner(rho0, xvec, xvec)
H1 = w * (a.dag() * a + 1 / 2)
h = (- (w * (a.dag() - a)) ** 2) / 8 + (((a.dag() + a) / alpha) ** 4) / (4 * w)

wmap = wigner_cmap(W_coherent)

fig, ax = plt.subplots(1, 1)
plot1 = ax.contourf(xvec, xvec, W_coherent, 100, cmap=wmap)
fig.colorbar(plot1, ax=ax)
ax.set_title("Random state in quartic potential")

tlist = np.linspace(0, 100, 1000)

# request that the solver return the expectation value of the photon number state operator
result = mesolve(h, rho0, tlist, [], [])


def animate_plot(i):
    plt.cla()
    W = wigner(result.states[i], xvec, xvec)
    ax.contourf(xvec, xvec, W, 100, cmap=wmap)
    ax.set_title("Random state in quartic potential")


ani = FuncAnimation(fig, animate_plot, frames=1000, interval=50)

# Save as gif

# ani.save('animation.gif', fps=10)

plt.show()
