import pennylane as qml
from pennylane import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy import linalg as la
import math


x = np.zeros(2**10)
x[0] = 1
# x[1] = 1
# x[-1] = 1

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
def grover_alg(input_list):
    M = np.count_nonzero(input_list)
    N = 2**num_n_wires
    theta = math.asin(2.0 * math.sqrt(M * (N - M)) / N)
    print("theta = ", theta)
    R = round(math.acos(math.sqrt(M / N)) / theta)
    print("R = ", R)

    for i in range(num_n_wires):
        qml.Hadamard(wires=i)

    for i in range(R):
        grover_operator()

    return qml.probs(wires=n_wires)
