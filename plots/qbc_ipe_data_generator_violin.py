import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qbc_ipe import *
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

num_t_wires_min = 2
num_t_wires_max = 7
num_t_wire_vals = num_t_wires_max - num_t_wires_min
num_t_wires_range = range(num_t_wires_min, num_t_wires_max)

num_reps = 50
t_wire_data = np.zeros((num_t_wire_vals, num_reps))

std = []
avg = []
num_shots = 10
data = np.zeros((num_t_wire_vals, num_shots * num_reps))
for t_idx, num_t_wires in enumerate(num_t_wires_range):
    t_data = []
    print("t_idx =", t_idx)
    for rep_idx in range(num_reps):
        x = 2 * np.random.random(2**2) - 1
        x = x / la.norm(x)
        y = 2 * np.random.random(2**2) - 1
        y = y / la.norm(y)
        rho, _, _, _, _, counts = qbc_ipe_algorithm(x, y, num_t_wires, num_shots)

        counts_vec = [
            [
                -(1.0 - 2.0 * np.sin(np.pi * int(key, 2) / 2**num_t_wires) ** 2)
                * la.norm(x)
                * la.norm(y)
                for key in counts.keys()
            ],
            list(counts.values()),
        ]
        rep_data = [
            counts_vec[0][i]
            for i in range(len(counts_vec[0]))
            for ctr in range(counts_vec[1][i])
        ]
        for val in rep_data:
            t_data.append(val - np.inner(x, y))

    data[t_idx, :] = t_data
    avg.append(np.mean(data[t_idx, :]))
    std.append(np.std(data[t_idx, :]))

df = pd.DataFrame(data)
df.to_csv("./data/qbc_ipe_data_violin.csv", header=None, index=None)
