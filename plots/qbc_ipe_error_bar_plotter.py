import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# data = np.genfromtxt("./data/qbc_ipe_data_error_bar.csv", delimiter=",").T
# print("data =", data)
# num_t_wire_vals = np.shape(data)[1]
# num_t_wire_min = 2
# num_t_wires_range = range(num_t_wire_min, num_t_wire_min + num_t_wire_vals)

# sns.set_theme(style="white")

# avg = np.mean(data, axis=0)
# std = np.std(data, axis=0)

# ax = sns.lineplot(x=num_t_wires_range, y=avg)
# ax.fill_between(num_t_wires_range, avg - 0.25 * std, avg + 0.25 * std, alpha=0.2)
# ax.set_yscale("log", base=2)
# ax.set_xlabel("Number of $t$ Wires")
# ax.set_ylabel("Absolute Error $\epsilon$")

# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# data = np.genfromtxt("./data/qbc_ipe_data_error_bar.csv", delimiter=",").T
# print("data =", data)
# num_t_wire_vals = np.shape(data)[1]
# num_t_wire_min = 2
# num_t_wires_range = range(num_t_wire_min, num_t_wire_min + num_t_wire_vals)

# sns.set_theme(style="darkgrid")

# avg = np.mean(data, axis=0)
# std = np.std(data, axis=0)

# ax = sns.lineplot(x=num_t_wires_range, y=avg, color="orange", marker="o", markersize=8)
# # ax.fill_between(
# #     num_t_wires_range, avg - 0.25 * std, avg + 0.25 * std, alpha=0.4, color="orange"
# # )
# ax.set_yscale("log", base=2)
# ax.set_xlabel("Number of $t$ Wires", fontsize=14)
# ax.set_ylabel("Absolute Error $\epsilon$", fontsize=14)
# ax.tick_params(labelsize=12)

# plt.title("Error vs Number of $t$ Wires for QBC-IPE Algorithm", fontsize=16)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = np.genfromtxt("./data/qbc_ipe_data_error_bar.csv", delimiter=",").T
print("data =", data)
num_t_wire_vals = np.shape(data)[1]
num_t_wire_min = 2
num_t_wires_range = range(num_t_wire_min, num_t_wire_min + num_t_wire_vals)

sns.set_theme(style="darkgrid")

avg = np.mean(data, axis=0)
std = np.std(data, axis=0)

# ax = sns.lineplot(x=num_t_wires_range, y=avg, yerr=0.25 * std, color="orange", marker="o", markersize=8) # comment out or delete this line
ax = sns.lineplot(
    x=num_t_wires_range,
    y=avg,
    err_style="bars",
    ci="sd",
    color="orange",
    marker="o",
    markersize=8,
)  # use this line instead
ax.set_yscale("log", base=2)
ax.set_xlabel("Number of $t$ Wires", fontsize=14)
ax.set_ylabel("Absolute Error $\epsilon$", fontsize=14)
ax.tick_params(labelsize=12)

plt.title("Error vs Number of $t$ Wires for QBC-IPE Algorithm", fontsize=16)
plt.show()
