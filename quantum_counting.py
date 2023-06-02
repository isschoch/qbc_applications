import pennylane as qml
from pennylane import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy import linalg as la


import pennylane as qml
from pennylane import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy import linalg as la
import math


x = [1, 0, 1, 0, 1, 0, 1, 1]
# marked_indices = np.array([3])

# for idx in marked_indices:
#     x[idx] = 3

num_n_wires = int(np.ceil(np.log2(len(x))))
num_t_wires = 10
t_wires = range(0, num_t_wires)
n_wires = range(num_t_wires, num_t_wires + num_n_wires)
o_wires = range(num_n_wires + num_t_wires, num_n_wires + num_t_wires + 1)
tot_wires = range(0, num_n_wires + num_t_wires + 1)
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
def quantum_counting(x):
    for i in n_wires:
        qml.Hadamard(wires=i)

    my_unitary = qml.matrix(grover_operator)()

    # OLD QPE #
    # qml.QuantumPhaseEstimation(
    #     grover_operator,
    #     estimation_wires=t_wires,
    #     target_wires=range(num_t_wires, num_t_wires + num_n_wires + 1),
    # )

    # NEW QPE #
    for t in range(num_t_wires):
        qml.Hadamard(wires=t)

    for idx, t in enumerate(t_wires):
        qml.ControlledQubitUnitary(
            np.linalg.matrix_power(my_unitary, 2 ** (num_t_wires - idx - 1)),
            control_wires=t,
            wires=range(num_t_wires, num_t_wires + num_n_wires + 1),
        )

    qml.adjoint(qml.QFT(wires=t_wires))

    return qml.probs(wires=t_wires)


result = quantum_counting(x)
print("result = ", result)
j = np.argmax(result)
theta = 2 * np.pi * j / 2**num_t_wires
f = np.sin(theta / 2) ** 2
rho = 2**num_n_wires * f
print("rho = {rho}, j = {j}".format(rho=rho, j=j))
