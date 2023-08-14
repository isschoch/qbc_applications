import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = np.genfromtxt("./data/qbc_ipe_data_violin.csv", delimiter=",").T
print("data =", data)
num_t_wire_vals = np.shape(data)[1]
num_t_wire_min = 2
num_t_wires_range = range(num_t_wire_min, num_t_wire_min + num_t_wire_vals)

sns.set_theme(style="white")
ax = sns.lineplot(x=range(num_t_wire_vals), y=np.zeros(num_t_wire_vals))
ax.lines[0].set_linestyle(":")
ax.lines[0].set_alpha(0.4)
ax.lines[0].set_color("black")
ax.lines[0].set_zorder(-100)
g = sns.violinplot(data=data, split=False, showextrema=False)
plt.setp(ax.collections, alpha=0.8)
plt.xlabel("Number of $t$ Wires")
plt.ylabel("$a - \mathbf{x}\cdot \mathbf{y}$")
g.set_xticks(range(num_t_wire_vals))
g.set_xticklabels(num_t_wires_range)
plt.savefig("./qbc_ipe_violin_plot.svg", dpi=300)


# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Load the data from a csv file and transpose it
# data = np.genfromtxt("./data/qbc_ipe_data_violin.csv", delimiter=",").T
# print("data =", data)

# # Set the range of the x-axis based on the number of t wires
# num_t_wire_vals = np.shape(data)[1]
# num_t_wire_min = 2
# num_t_wires_range = range(num_t_wire_min, num_t_wire_min + num_t_wire_vals)

# # Set the theme and style of the plot
# sns.set_theme(style="white")
# sns.set_style("ticks")

# # Create a figure and an axis object with some padding
# fig, ax = plt.subplots()
# fig.tight_layout(pad=2)

# # Plot a horizontal dashed line at y=0 with low opacity and black color
# ax.axhline(y=0, ls=":", alpha=0.4, color="black", zorder=-100)

# # Plot a violin plot of the data with no split and no extrema
# # Use a custom color palette and adjust the width and inner lines of the violins
# palette = sns.color_palette("mako", num_t_wire_vals)
# g = sns.violinplot(
#     data=data,
#     split=False,
#     showextrema=False,
#     ax=ax,
#     palette=palette,
#     width=0.8,
#     inner="quartile",
# )

# # Adjust the alpha value of the violins
# plt.setp(ax.collections, alpha=0.8)

# # Set the labels of the x-axis and y-axis with LaTeX formatting
# plt.xlabel("Number of $t$ Wires")
# plt.ylabel("$a - \mathbf{x}\cdot \mathbf{y}$")

# # Set the ticks and labels of the x-axis based on the range of t wires
# g.set_xticks(range(num_t_wire_vals))
# g.set_xticklabels(num_t_wires_range)

# # Remove the top and right spines of the plot
# sns.despine()

# # Save the plot as an svg file with high resolution
# # plt.savefig("./qbc_ipe_violin_plot.svg", dpi=300)
# plt.show()
