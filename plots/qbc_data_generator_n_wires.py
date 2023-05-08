import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qbc import *
import math
import matplotlib.pyplot as plt
import pandas as pd

num_n_wires_min = 2
num_n_wires_max = 8
num_n_wire_vals = num_n_wires_max - num_n_wires_min
num_n_wires_range = range(num_n_wires_min, num_n_wires_max)

num_reps = 20
n_wire_data = np.zeros((num_n_wire_vals, num_reps))

for (n_wire_idx, num_n_wires) in enumerate(num_n_wires_range):
    print("num_n_wires =", num_n_wires)
    for rep_idx in range(num_reps):
        x = np.random.randint(0, 2, size=(2**num_n_wires))
        y = np.random.randint(0, 2, size=(2**num_n_wires))
        rho, _, _, _, _ = qbc_algorithm(x, y, num_n_wires)
        error = abs(rho - np.inner(x, y))
        n_wire_data[n_wire_idx, rep_idx] = error

df = pd.DataFrame(n_wire_data)
df.insert(0, 'new_col', num_n_wires_range)
df.to_csv("./data/qbc_data_n_wires.csv", header=None, index=None)
