import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc


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

# New Data
new_data = [
    [(-7.516378266470955, -3.516378266470955), (1.322783566875362, 5.322783566875362), 4.920986222694281],
    [(-7.516378266470955, -3.516378266470955), (1.322783566875362, 5.322783566875362), 4.203664097873678],
    [(0.2642932849371107, 4.264293284937111), (8.71550872023414, 10.0), 1.5855000560674484],
    [(0.2642932849371107, 4.264293284937111), (8.71550872023414, 10.0), 1.4301983115435615],
    [(2.0751135846725877, 6.075113584672588), (-7.72649391433565, -3.7264939143356504), 3.9148082742998396],
    [(2.0751135846725877, 6.075113584672588), (-7.72649391433565, -3.7264939143356504), 3.1040195627109513],
    [(8.00225613325625, 10.0), (-0.5236497685229029, 3.476350231477097), 6.056228479009957],
    [(8.00225613325625, 10.0), (-0.5236497685229029, 3.476350231477097), 5.966257518403349],
    [(1.234906455897491, 5.234906455897491), (7.353635606376315, 10.0), 2.6804040389633044],
    [(1.234906455897491, 5.234906455897491), (7.353635606376315, 10.0), 2.5678080287621095],
    [(2.6440965160138497, 6.64409651601385), (-9.324235799570557, -5.324235799570557), 4.402974449616318],
    [(2.6440965160138497, 6.64409651601385), (-9.324235799570557, -5.324235799570557), 3.1041668132158686],
    [(-4.689090065412122, -0.6890900654121221), (0.7591980569207628, 4.759198056920763), 3.282657029019573],
    [(-4.689090065412122, -0.6890900654121221), (0.7591980569207628, 4.759198056920763), 3.014765066439228],
    [(-3.0332021067632553, 0.9667978932367447), (-7.890760398127008, -3.890760398127008), 4.775073804857068],
    [(-3.0332021067632553, 0.9667978932367447), (-7.890760398127008, -3.890760398127008), 4.0276771506317814],
    [(-5.145796466281656, -1.1457964662816558), (-1.8179917885037433, 2.1820082114962567), 8.524773318584167],
    [(-5.145796466281656, -1.1457964662816558), (-1.8179917885037433, 2.1820082114962567), 4.460158118375638],
    [(-1.538423614656308, 2.461576385343692), (-0.5074773671398489, 3.492522632860151), 3.5658393410854377],
    [(-1.538423614656308, 2.461576385343692), (-0.5074773671398489, 3.492522632860151), 2.3267569400401418]
]

# Plotting the regions as boxes
fig, ax = plt.subplots(figsize=(4, 4))

# Normalize disagreement scores for color mapping
norm = plt.Normalize(min(d[2] for d in new_data), max(d[2] for d in new_data))
cmap = plt.get_cmap('coolwarm')

for region in new_data:
    (x1, x2), (y1, y2), score = region
    width = x2 - x1
    height = y2 - y1
    color = cmap(norm(score))
    rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

# Set the limits of the plot to ensure all boxes are visible
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Adding color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Uncertainty Score')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
# plt.title('Disagreement Score Heatmap with Regions as Boxes')
plt.grid(True)
show=False
if show:
    plt.show()
else:
    fname = os.path.join("uncertainty_region_width_{}.pdf".format(0.1))
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()