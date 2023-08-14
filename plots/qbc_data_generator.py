import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qbc.qbc import *
import math
import matplotlib.pyplot as plt
import pandas as pd

n = 5
num_t_wires_min = 2
num_t_wires_max = 12
num_t_wire_vals = num_t_wires_max - num_t_wires_min
num_t_wires_range = range(num_t_wires_min, num_t_wires_max)

num_reps = 100
t_wire_data = np.zeros((num_t_wire_vals, num_reps))

for t_wire_idx, num_t_wires in enumerate(num_t_wires_range):
    for rep_idx in range(num_reps):
        x = np.random.randint(0, 2, size=(2**n))
        y = np.random.randint(0, 2, size=(2**n))
        rho, _, _, _, _ = qbc_algorithm(x, y, num_t_wires)
        error = abs(rho - np.inner(x, y))
        t_wire_data[t_wire_idx, rep_idx] = error

df = pd.DataFrame(t_wire_data)
df.insert(0, "new_col", num_t_wires_range)
df.to_csv("./data/qbc_data.csv", header=None, index=None)
