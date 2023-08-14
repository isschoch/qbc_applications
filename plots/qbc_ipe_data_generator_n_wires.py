import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipe.qbc_ipe import *
import math
import matplotlib.pyplot as plt
import pandas as pd

num_n_wires_min = 2
num_n_wires_max = 8
num_n_wire_vals = num_n_wires_max - num_n_wires_min
num_n_wires_range = range(num_n_wires_min, num_n_wires_max)

num_reps = 20
n_wire_data = np.zeros((num_n_wire_vals, num_reps))

for n_wire_idx, num_n_wires in enumerate(num_n_wires_range):
    print("num_n_wires =", num_n_wires)
    for rep_idx in range(num_reps):
        x = np.random.randint(0, 2, size=(2**num_n_wires))
        y = np.random.randint(0, 2, size=(2**num_n_wires))
        rho, _, _, _, _, counts = qbc_ipe_algorithm(x, y, num_n_wires)
        error = abs(rho - np.inner(x, y))
        n_wire_data[n_wire_idx, rep_idx] = error

df = pd.DataFrame(n_wire_data)
df.insert(0, "new_col", num_n_wires_range)
df.to_csv("./data/qbc_ipe_data_n_wires.csv", header=None, index=None)


# x = np.array([1.0, 1.0, 0.0, 1.0])
# y = np.array([1.0, 1.0, 1.0, 1.0])


# # print(test_ipe(x, y))
# rho, f, theta, j, probs, counts = qbc_ipe_algorithm(x, y)
# print("j =", j)
# print("theta =", theta)
# print("rho =", rho)
# print("counts =", counts)


# print("np.inner(x, y) =", np.inner(x, y))

# rho, f, theta, j, probs, counts = qbc_ipe_algorithm(x, y)
# print("j =", j)
# print("theta =", theta)
# print("rho =", rho)
# print("counts =", counts)

# plt.plot(counts.keys(), counts.values())
# plt.show()

# # print(y / la.norm(y))
# # print(1.0 / np.sqrt(2) * y / la.norm(y))

# a = -5.0
# b = 5.0
# dim = 8
# for i in range(10):
#     x = (b - a) * np.random.rand(dim) + a
#     y = (b - a) * np.random.rand(dim) + a
#     rho, f, theta, j, probs, counts = qbc_ipe_algorithm(x, y)
#     print("j =", j)
#     print("theta =", theta)
#     print("rho =", rho)
#     # print(
#     #     "analytical result = ", 0.5 * (1.0 - np.inner(x, y) / (la.norm(x) * la.norm(y)))
#     # )
#     print("np.inner(x, y) =", np.inner(x, y))
