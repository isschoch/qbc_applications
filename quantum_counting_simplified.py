import pennylane as qml
from pennylane import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy import linalg as la
import math


x = np.zeros(2**10)
marked_indices = np.array([0, 3, -1])

for idx in marked_indices:
    x[idx] = 1

num_n_wires = int(np.ceil(np.log2(len(x))))
n_wires = range(0, num_n_wires)
o_wires = range(num_n_wires, num_n_wires + 1)
tot_wires = range(0, num_n_wires + 1)
dev = qml.device("default.qubit", wires=tot_wires)
x_indices = [
    [int(k) for k in format(elem, "0%sb" % num_n_wires)] for elem in np.nonzero(x)[0]
]


# Oracle A encoding x data
def U_x():
    for one_idx in x_indices:
        qml.ctrl(qml.PauliX, control=n_wires, control_values=one_idx)(wires=o_wires)


def phase_oracle():
    U_x()
    qml.PauliZ(wires=o_wires)
    U_x()


def grover_operator():
    phase_oracle()
    qml.GroverOperator(wires=n_wires)


@qml.qnode(dev)
def grover_alg(input_list, num_iterations):
    for i in range(num_n_wires):
        qml.Hadamard(wires=i)

    for i in range(num_iterations):
        grover_operator()

    return qml.probs(wires=n_wires)


def lemma_2(theta_min, theta_max):
    delta_theta = theta_max - theta_min
    k = round(theta_min / (2 * delta_theta))
    r = round(math.pi * k / theta_min)
    if r % 2 == 0:
        r -= 1

    return r


def simplified_quantum_counting(x):
    theta = math.asin(math.sqrt(np.count_nonzero(x) / 2**num_n_wires))
    print("theta = ", theta)
    success_rate = 0.0
    k = 0
    r_k_prev = 0
    epsilon = 0.1
    gamma = 0.5
    while success_rate < 0.95:
        k += 1
        r_k = math.floor(1.05**k)
        if r_k % 2 == 0:
            r_k -= 1
        assert r_k % 2 == 1
        print("k = {k}, r_k = {r_k}".format(k=k, r_k=r_k))

        if r_k_prev != r_k:
            result = grover_alg(x, (r_k - 1) // 2)
            tmp_sum = 0.0
            for idx in marked_indices:
                tmp_sum += result[idx]
            success_rate = tmp_sum

        r_k_prev = r_k
    k_end = k
    print(
        "k_end = {k_end}, success_rate = {success_rate}".format(
            k_end=k_end, success_rate=success_rate
        )
    )

    theta_min = 0.9 * 1.05 ** (-k_end)
    additional_factor = 2.0
    theta_max = 1.65 * theta_min * additional_factor

    success_rate = 0.0
    t = 0
    r_t_prev = 0
    while theta_max > (1 + epsilon / 5) * theta_min:
        r_t = lemma_2(theta_min, theta_max)
        print(
            "t = {t}, r_t = {r_t}, success_rate = {success_rate}".format(
                t=t, r_t=r_t, success_rate=success_rate
            )
        )
        print(
            "theta_min = {theta_min}, theta_max = {theta_max}, (1 + epsilon / 5) * theta_min = {val}".format(
                theta_min=theta_min,
                theta_max=theta_max,
                val=(1 + epsilon / 5) * theta_min,
            )
        )
        if r_t != r_t_prev:
            result = grover_alg(x, (r_t - 1) // 2)
            tmp_sum = 0.0
            for idx in marked_indices:
                tmp_sum += result[idx]
            success_rate = tmp_sum

        gamma = theta_max / theta_min - 1.0
        if success_rate >= 0.12:
            theta_min = theta_max / (1 + 0.9 * gamma)
        else:
            theta_max = theta_min * (1 + 0.9 * gamma)

        r_t_prev = r_t
        t += 1

    K = 2**num_n_wires * math.sin(theta_max) ** 2
    return K


result = simplified_quantum_counting(x)
print("result = ", result)
