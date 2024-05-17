import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from phaseportrait import PhasePortrait2D, PhasePortrait3D
import matplotlib.pyplot as plt
import os

palette = ['#ff3b30', '#4cd964', '#ffcc00', '#007aff', '#5856d6', '#ff9500', '#5ac8fa', '#ff2d55', '#969783']
markers = ['o', '^', '<', 'd', '*', '2']
sns.set_style("ticks")

mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.linewidth'] = 1
rc("font", **{'family': 'serif', 'serif': ['Palatino'], 'size': 16})
rc('text', usetex=True)
plt.style.use("seaborn-v0_8-bright")

eq_idx = 'vars1_prog10'


def pred_by_act_ode(x0, dx0):
    x=[x0]
    return x0, -0.8*x[0]*(1 - x[0])**1.2 + 0.2*x[0]**1.2 - 0.2*x[0]**2.2





fig = plt.figure(figsize=(4, 3))
Oscillator1 = PhasePortrait2D(pred_by_act_ode, Range=[0.1, 0.9], MeshDim=21, Title="",
                              xlabel=r"$\Theta$", ylabel=r"$\dot{\Theta}$", fig=fig, )
fig, ax = Oscillator1.plot(color="grey")
# fig.suptitle("Predicted ODE $\phi_1=(10\sin(x_2),  4\cos(x_1+2))$", fontsize=13)
plt.show()
#
# fname = os.path.join(eq_idx + "_phase.correct.pdf")
# plt.savefig(fname, bbox_inches='tight', pad_inches=0)
# plt.close()
