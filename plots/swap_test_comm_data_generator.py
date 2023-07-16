import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qbc_ipe import *
from swap_test import *
import math
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    num_shots_range = []
    for i in range(2, 10):
        num_shots_range.append(2**i)
    num_n_wires = 2
    num_shots_range = num_shots_range

    num_reps = 100

    data = np.zeros(shape=(len(num_shots_range), num_reps))
    for idx, num_shots in enumerate(num_shots_range):
        # TODO: make this consistent with the qbc_ipe case
        for rep_idx in range(num_reps):
            x = 2 * np.random.random(2**num_n_wires)
            x = x / la.norm(x)
            y = 2 * np.random.random(2**num_n_wires)
            y = y / la.norm(y)
            print("num_shots =", num_shots, "rep_idx =", rep_idx)
            inner_prod, probs, counts = swap_test_algorithm(x, y, num_shots=num_shots)
            data[idx, rep_idx] = abs(inner_prod - np.inner(x, y))

    df = pd.DataFrame(data)
    num_comms = np.array(num_shots_range) * num_n_wires
    df.insert(0, "new_col", num_comms)
    df.to_csv("./data/swap_test_comm.csv", header=None, index=None)
