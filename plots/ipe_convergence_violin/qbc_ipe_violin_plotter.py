import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.figure(figsize=(8, 7))
data = np.genfromtxt("qbc_ipe_data_violin.csv", delimiter=",").T
print("data =", data)
num_t_wire_vals = np.shape(data)[1]
num_t_wire_min = 2
num_t_wires_range = range(num_t_wire_min, num_t_wire_min + num_t_wire_vals)

palette = sns.color_palette("hls", num_t_wire_vals)
sns.set_theme(style="white", font_scale=2.2, palette=palette)
ax = sns.lineplot(
    x=range(num_t_wire_vals),
    y=np.zeros(num_t_wire_vals),
    linestyle=":",
    alpha=0.6,
    color="black",
    linewidth=3,
)
ax.lines[0].set_zorder(-100)
g = sns.violinplot(
    data=data,
    split=False,
    showextrema=False,
    linewidth=2.5,
    width=1.0,
)
# g = sns.boxplot(data=data, linewidth=1.75)
plt.yticks(np.arange(-2.0, 3.0, 1.0))

plt.setp(ax.collections, alpha=0.8)
plt.xlabel("Number of $t$ Wires")
plt.ylabel("$a - \mathbf{x}\cdot \mathbf{y}$")
g.set_xticks(range(num_t_wire_vals))
g.set_xticklabels(num_t_wires_range)
plt.tight_layout()
# plt.show()
plt.savefig("./qbc_ipe_violin_plot.png", dpi=400)
