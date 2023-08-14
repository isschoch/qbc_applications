import sys
import os

dir_path = "/Users/isidorschoch/Programs/master_thesis/qbc_applications/"
sys.path.append(dir_path)
from ipe.qbc_ipe import *

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    num_t_wires_range = np.arange(1, 10)

    num_n_wires = 2
    num_reps = 100
    num_shots = 1
    data = np.zeros(shape=(len(num_t_wires_range), num_reps))
    for idx, num_t_wires in enumerate(num_t_wires_range):
        # TODO: make this consistent with the qbc_ipe case
        for rep_idx in range(num_reps):
            x = 2 * np.random.random(2**num_n_wires)
            x = x / la.norm(x)
            y = 2 * np.random.random(2**num_n_wires)
            y = y / la.norm(y)
            inner_prod, _, _, _, _, _ = qbc_ipe_algorithm(
                x, y, num_t_wires=num_t_wires, num_shots=num_shots
            )
            data[idx, rep_idx] = abs(inner_prod - np.inner(x, y))
            if rep_idx % 100 == 0:
                print("rep_idx =", rep_idx, "num_t_wires =", num_t_wires)

    df = pd.DataFrame(data)
    num_comms = (2**num_t_wires_range - 1) * num_n_wires
    df.insert(0, "new_col", num_comms)
    df.to_csv("qbc_ipe_comm.csv", header=None, index=None)
