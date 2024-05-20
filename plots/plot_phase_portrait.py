import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from phaseportrait import PhasePortrait2D, PhasePortrait3D, Trajectory3D
import matplotlib.pyplot as plt
from scibench.data import equation_object_loader
import os

palette = ['#ff3b30', '#4cd964', '#ffcc00', '#007aff', '#5856d6', '#ff9500', '#5ac8fa', '#ff2d55', '#969783']
markers = ['o', '^', '<', 'd', '*', '2']
sns.set_style("ticks")

mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['lines.linewidth'] = 1
rc("font", **{'family': 'serif', 'serif': ['Palatino'], 'size': 16})
rc('text', usetex=True)
plt.style.use("seaborn-v0_8-bright")


# %%
def replace_math_operator(title):
    title = "$" + ",".join(title) + "$"
    title = title.replace('**', '^')
    title = title.replace('[0]', '_1')
    title = title.replace('[1]', '_2')
    title = title.replace('[2]', '_3')
    title = title.replace('*', '')
    title = title.replace('np.sin', '\sin')
    title = title.replace('np.cos', '\cos')
    title = title.replace('np.cot', '\cot')
    return title


for i in range(10, 11):
    name = f"vars2_prog{i}"
    true_equation = equation_object_loader(name)
    ranged = [xi.range for xi in true_equation.vars_range_and_types]
    print(name, ranged, true_equation)
    func = lambda x0, x1: true_equation.np_eq(t=None, x=[x0, x1]).tolist()

    fig = plt.figure(figsize=(3, 3))
    Oscillator1 = PhasePortrait2D(func, Range=ranged, MeshDim=21, Title="", xlabel=r"$x_1$", ylabel=r"$x_2$", fig=fig)
    fig, ax = Oscillator1.plot(color="grey")
    title = [r"\dot{x}_" + str(i + 1) + "=" + one_str for i, one_str in enumerate(true_equation.sympy_eq)]
    title = f"Equation ID {i}" + ": " + true_equation._description
    print(title)
    fig.suptitle(title, fontsize=11)
    # plt.show()
    #
    fname = os.path.join(name + "_phase.correct.pdf")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()

# %%
for i in range(1, 11):
    name = f"vars3_prog{i}"
    true_equation = equation_object_loader(name)
    ranged = [xi.range for xi in true_equation.vars_range_and_types]
    print(name, ranged, true_equation)
    func = lambda x0, x1, x2: true_equation.np_eq(t=None, x=[x0, x1, x2]).tolist()

    # fig, ax = plt.figure(figsize=(3, 3)).add_subplot(projection='3d')
    Oscillator1 = PhasePortrait3D(func, Range=ranged,MeshDim=6, n_points=20000, Title="",
                                  xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$")
    fig, ax = Oscillator1.plot(color="grey")
    # title = [r"\dot{x}_" + str(i+1) + "=" + one_str for i, one_str in enumerate(true_equation.sympy_eq)]
    # title=replace_math_operator(title)
    title = f"Equation ID {i}" + ": " + true_equation._description
    print(title)
    fig.suptitle(title, fontsize=11)
    fname = os.path.join(name + "_phase.correct.pdf")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()
