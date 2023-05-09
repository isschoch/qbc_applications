import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("./data/qbc_data.csv", delimiter=",")
print("data =", data)
num_t_wires = data[:, 0]
errors = data[:, 1:]
print("num_t_wires =", num_t_wires)
print("times =", errors)
errors_avg = np.mean(errors, axis=1)
errors_std = np.std(errors, axis=1)

# plt.yscale('log', base=2)
# plt.plot(num_t_wires, errors_avg, 'o--')
# plt.fill_between(num_t_wires, errors_avg - errors_std, errors_avg + errors_std, alpha=0.2)

plt.yscale("log", base=2)
plt.xlabel("Number of t wires")
# plt.errorbar(num_t_wires, errors_avg, yerr=errors_std, fmt='o')
plt.plot(num_t_wires, errors_avg, "o--")
plt.ylabel("Average Error")
plt.title("Error vs. Number of t wires")
# plt.show()
plt.savefig("./qbc_data_plot.png", dpi=300)
print("errors_avg =", errors_avg)
print("errors_std =", errors_std)
