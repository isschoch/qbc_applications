import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("./data/qbc_data_n_wires.csv", delimiter=",")
print("data =", data)
num_n_wires = data[:, 0]
errors = data[:, 1:]
print("num_t_wires =", num_n_wires)
print("times =", errors)
errors_avg = np.mean(errors, axis=1)
errors_std = np.std(errors, axis=1)

# plt.yscale('log', base=2)
# plt.plot(num_t_wires, errors_avg, 'o--')
# plt.fill_between(num_t_wires, errors_avg - errors_std, errors_avg + errors_std, alpha=0.2)

plt.yscale("log", base=2)
plt.plot(num_n_wires, errors_avg, "o--")
plt.ylabel("Average Error")
plt.title("Error vs. Number of n wires")
plt.savefig("./qbc_data_plot_n_wires.png", dpi=300)

print("errors_avg =", errors_avg)
print("errors_std =", errors_std)
