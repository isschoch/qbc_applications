import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np

if __name__ == "__main__":
    st_data = np.genfromtxt("./data/swap_test_comm.csv", delimiter=",")
    ipe_data = np.genfromtxt("./data/qbc_ipe_comm.csv", delimiter=",")

    st_errors = st_data[:, 1:]
    st_num_shots_range = st_data[:, 0]
    avg_st_errors = np.mean(st_errors, axis=1)
    std_st_errors = np.std(st_errors, axis=1)

    ipe_errors = ipe_data[:, 1:]
    ipe_num_shots_range = ipe_data[:, 0]
    avg_ipe_errors = np.mean(ipe_errors, axis=1)
    std_ipe_errors = np.std(ipe_errors, axis=1)

    st_coeff = np.polyfit(np.log(st_num_shots_range), np.log(avg_st_errors), deg=1)
    print("coeff =", st_coeff)
    f, ax = plt.subplots(figsize=(7, 7))
    sns.set_palette("bright")
    ax.set(xscale="log", yscale="log")
    print("st_num_shots_range =", st_num_shots_range)
    print("avg_st_errors =", avg_st_errors)

    def lin_approx(x, coeff):
        return coeff[0] * x + coeff[1]

    x_start_st = st_num_shots_range[1] / 1.5
    x_end_st = x_start_st * 10
    y_shift_st = 0.1
    y_start_st = np.exp(lin_approx(np.log(x_start_st), st_coeff) - y_shift_st)
    y_end_st = np.exp(lin_approx(np.log(x_end_st), st_coeff) - y_shift_st)

    slope_triangle = Polygon(
        [[x_start_st, y_start_st], [x_start_st, y_end_st], [x_end_st, y_end_st]], True
    )
    p_st = PatchCollection([slope_triangle], alpha=0.5, color="indianred")
    ax.add_collection(p_st)
    ax.text(x_start_st * (x_end_st / x_start_st) ** 0.4, y_end_st + 0.001, r"$10^{1}$")
    st_coeff_string = str(st_coeff[0])[0:6]
    ax.text(
        x_start_st,
        y_start_st * (y_end_st / y_start_st) ** 0.6,
        r"$10^{" + st_coeff_string + "}$",
    )

    ax.set_xlim(
        [
            min(st_num_shots_range[0], ipe_num_shots_range[0]),
            max(st_num_shots_range[-1], ipe_num_shots_range[-1]),
        ]
    )
    ax.set_xlabel("Number of Communication Rounds")
    ax.set_ylabel("Error $\epsilon$")

    ipe_coeff = np.polyfit(np.log(ipe_num_shots_range), np.log(avg_ipe_errors), deg=1)
    x_start_ipe = ipe_num_shots_range[3] * 1.5
    x_end_ipe = x_start_ipe * 10
    y_shift_ipe = 0.1
    y_start_ipe = np.exp(lin_approx(np.log(x_start_ipe), ipe_coeff) - y_shift_ipe)
    y_end_ipe = np.exp(lin_approx(np.log(x_end_ipe), ipe_coeff) - y_shift_ipe)

    slope_triangle = Polygon(
        [[x_start_ipe, y_start_ipe], [x_start_ipe, y_end_ipe], [x_end_ipe, y_end_ipe]],
        True,
    )
    p_ipe = PatchCollection([slope_triangle], alpha=0.5, color="teal")
    ax.add_collection(p_ipe)
    ax.text(
        x_start_ipe * (x_end_ipe / x_start_ipe) ** 0.4, y_end_ipe + 0.0001, r"$10^{1}$"
    )
    ipe_coeff_string = str(ipe_coeff[0])[0:6]
    ax.text(
        x_start_ipe,
        y_start_ipe * (y_end_ipe / y_start_ipe) ** 0.6,
        r"$10^{" + ipe_coeff_string + "}$",
    )

    sns.lineplot(
        x=st_num_shots_range,
        y=avg_st_errors,
        ax=ax,
        marker="o",
        linestyle=":",
        color="darkred",
    )
    ax.fill_between(
        st_num_shots_range,
        avg_st_errors - std_st_errors,
        avg_st_errors + std_st_errors,
        alpha=0.1,
        color="salmon",
    )

    sns.lineplot(
        x=ipe_num_shots_range,
        y=avg_ipe_errors,
        ax=ax,
        marker="o",
        linestyle=":",
        color="darkslategrey",
    )
    ax.fill_between(
        ipe_num_shots_range,
        avg_ipe_errors - std_ipe_errors,
        avg_ipe_errors + std_ipe_errors,
        alpha=0.1,
        color="darkturquoise",
    )
    sns.set_palette("bright")
    plt.legend(
        [
            "Swap Test",
            "Swap Test Slope",
            "CI Swap Test",
            "IPE",
            "CI IPE Slope",
            "CI IPE",
        ]
    )
    plt.show()
