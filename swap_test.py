import pennylane as qml
from pennylane import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def swap_test_algorithm(x, y, num_shots=10000):
    assert len(x) == len(y)

    num_n_wires = int(np.ceil(np.log2(len(x))))
    num_m_wires = num_n_wires
    num_tot_wires = num_n_wires + num_n_wires + 1

    n_wires = range(0, num_n_wires)
    m_wires = range(num_n_wires, num_n_wires + num_n_wires)
    a_wires = range(num_n_wires + num_n_wires, num_n_wires + num_n_wires + 1)
    tot_wires = range(0, num_n_wires + num_n_wires + 1)

    dev = qml.device("default.qubit", wires=tot_wires, shots=num_shots)

    A_x = np.zeros((2**num_n_wires, 2**num_n_wires))
    A_x[:, 0] = np.array(x)
    Q_x, _ = la.qr(A_x, mode="full")
    for i in range(2**num_n_wires):
        Q_x[i] /= la.norm(Q_x[i])

    u_x = Q_x.T
    if np.sign(u_x[0, 0]) != np.sign(x[0]):
        u_x *= -1

    # Oracle A encoding x data
    def U_x():
        qml.QubitUnitary(u_x, wires=n_wires)

    A_y = np.zeros((2**num_m_wires, 2**num_m_wires))
    A_y[:, 0] = np.array(y)
    Q_y, _ = la.qr(A_y, mode="full")
    for i in range(2**num_m_wires):
        Q_y[i] /= la.norm(Q_y[i])

    u_y = Q_y.T
    if np.sign(u_y[0, 0]) != np.sign(y[0]):
        u_y *= -1

    # Oracle B encoding y data
    def U_y():
        qml.QubitUnitary(u_y, wires=m_wires)

    # QBC circuit
    @qml.qnode(dev, interface=None)
    def swap_test():
        U_x()
        U_y()
        # qml.AmplitudeEmbedding(x, wires=n_wires, normalize=True)
        # qml.AmplitudeEmbedding(y, wires=m_wires, normalize=True)
        qml.Hadamard(wires=a_wires)
        for i in range(num_n_wires):
            qml.CSWAP(wires=[a_wires[0], n_wires[i], m_wires[i]])
        qml.Hadamard(wires=a_wires)

        return qml.probs(wires=a_wires), qml.counts(wires=a_wires)

    probs, counts = swap_test()

    if probs[0] < 0.5:
        probs[0] = 0.5

    inner_prod = np.sqrt(2 * probs[0] - 1.0) * la.norm(x) * la.norm(y)
    return inner_prod, probs, counts


# num_shots_range = []
# for i in range(1, 10):
#     num_shots_range.append(2**i)
# print(num_shots_range)
# error = np.zeros(len(num_shots_range))
# num_n_wires = 2
# num_reps = 100
# errors = np.zeros(shape=(len(num_shots_range), num_reps))
# for idx, num_shots in enumerate(num_shots_range):
#     # TODO: make this consistent with the qbc_ipe case
#     for rep_idx in range(num_reps):
#         x = 2 * np.random.random(2**num_n_wires)
#         x = x / la.norm(x)
#         y = 2 * np.random.random(2**num_n_wires)
#         y = y / la.norm(y)
#         inner_prod, probs, counts = swap_test_algorithm(x, y, num_shots)
#         errors[idx, rep_idx] = abs(inner_prod - np.inner(x, y))

# print(errors.shape)
# avg_errors = np.mean(errors, axis=1)
# std_errors = np.std(errors, axis=1)


# coeff = np.polyfit(np.log(num_shots_range), np.log(avg_errors), deg=1)
# print("coeff =", coeff)
# f, ax = plt.subplots(figsize=(7, 7))
# ax.set(xscale="log", yscale="log")
# print("num_shots_range =", num_shots_range)
# print("avg_errors =", avg_errors)


# def lin_approx(x):
#     return coeff[0] * x + coeff[1]


# x_start = num_shots_range[2]
# x_end = x_start * 10
# y_shift = 0.1
# y_start = np.exp(lin_approx(np.log(x_start)) - y_shift)
# y_end = np.exp(lin_approx(np.log(x_end)) - y_shift)

# slope_triangle = Polygon([[x_start, y_start], [x_start, y_end], [x_end, y_end]], True)
# p = PatchCollection([slope_triangle], alpha=0.5, color="black")
# ax.add_collection(p)
# ax.text(x_start * (x_end / x_start) ** 0.4, y_end + 0.001, r"$10^{1}$")
# coeff_string = str(coeff[0])[0:6]
# ax.text(x_start, y_start * (y_end / y_start) ** 0.6, r"$10^{" + coeff_string + "}$")
# ax.set_xlim([num_shots_range[0], num_shots_range[-1]])
# ax.set_xlabel("Number of Communication Rounds")
# ax.set_ylabel("Error $\epsilon$")

# ax.fill_between(
#     num_shots_range, avg_errors - std_errors, avg_errors + std_errors, alpha=0.25
# )
# sns.lineplot(x=num_shots_range, y=avg_errors, ax=ax, marker="o", linestyle=":")
# plt.show()
