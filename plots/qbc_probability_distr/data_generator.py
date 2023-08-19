import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

min_num_t_wires = 5
max_num_t_wires = 9

t_wires_range = range(min_num_t_wires, max_num_t_wires + 1, 2)
data = []
for num_t_wires in t_wires_range:
    T = 2**num_t_wires
    theta = np.pi / np.exp(1)
    probs = (
        1.0
        / (2 * T**2)
        * np.array(
            [
                np.sin(T * theta / 2 - np.pi * j) ** 2
                / np.sin(theta / 2 - np.pi * j / T) ** 2
                + np.sin(T * theta / 2 + np.pi * j) ** 2
                / np.sin(theta / 2 + np.pi * j / T) ** 2
                for j in range(T)
            ]
        )
    )
    data.append(probs)

df = pd.DataFrame(data)
df.index = t_wires_range
df.to_csv("data.csv", header=True, index=True)
