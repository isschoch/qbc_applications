import pennylane as qml
from pennylane import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy import linalg as la


def qbc_algorithm(x, y, num_t_wires=10):
    assert len(x) == len(y)
    num_n_wires = int(np.ceil(np.log2(len(x))))

    t_wires = range(0, num_t_wires)
    n_wires = range(num_t_wires, num_t_wires + num_n_wires)
    o_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + 2)
    tot_wires = range(0, num_t_wires + num_n_wires + 2)

    dev = qml.device("default.qubit", wires=tot_wires)

    x_indices = [
        [int(k) for k in format(elem, "0%sb" % num_n_wires)]
        for elem in np.nonzero(x)[0]
    ]
    # print("x =", x)
    # print("x_indices =", x_indices)

    y_indices = [
        [int(k) for k in format(elem, "0%sb" % num_n_wires)]
        for elem in np.nonzero(y)[0]
    ]
    # print("y =", y)
    # print("y_indices =", y_indices)

    # Oracle A encoding x data
    def U_x():
        for one_idx in x_indices:
            qml.ctrl(qml.PauliX, control=n_wires, control_values=one_idx)(
                wires=o_wires[0]
            )

    # Oracle B encoding y data
    def U_y():
        for one_idx in y_indices:
            qml.ctrl(qml.PauliX, control=n_wires, control_values=one_idx)(
                wires=o_wires[1]
            )

    # Phase oracle used in Grover operator
    def phase_oracle(U_x, U_y):
        U_x()
        U_y()
        qml.CZ(wires=[o_wires[0], o_wires[1]])
        U_y()
        U_x()

    # Grover operator used in QPE
    def grover_operator(U_x, U_y):
        phase_oracle(U_x, U_y)
        qml.GroverOperator(wires=n_wires)

    # QBC circuit
    @qml.qnode(dev)
    def qbc(U_x, U_y):
        for i in n_wires:
            qml.Hadamard(wires=i)
        my_unitary = qml.matrix(grover_operator)(U_x, U_y)
        qml.QuantumPhaseEstimation(
            my_unitary,
            target_wires=range(num_t_wires, num_t_wires + num_n_wires + 2),
            estimation_wires=t_wires,
        )
        return qml.probs(wires=t_wires)

    # Helper fct for debugging U_x oracle
    @qml.qnode(dev)
    def U_x_vec(i):
        qml.BasisEmbedding(i, wires=n_wires)
        U_x()
        return qml.probs(wires=o_wires[0])

    def U_x_vec_val(i):
        return np.argmax(U_x_vec(i))

    def U_x_vec_full():
        res = [U_x_vec_val(i) for i in range(2**num_n_wires)]
        return res

    # Helper fct for debugging U_y oracle
    @qml.qnode(dev)
    def U_y_vec(i):
        qml.BasisEmbedding(i, wires=n_wires)
        U_y()
        return qml.probs(wires=o_wires[1])

    def U_y_vec_val(i):
        return np.argmax(U_y_vec(i))

    def U_y_vec_full():
        res = [U_y_vec_val(i) for i in range(2**num_n_wires)]
        return res

    probs = qbc(U_x, U_y)
    j = np.argmax(probs)
    theta = 2 * np.pi * j / 2**num_t_wires
    f = np.sin(theta / 2) ** 2
    rho = 2**num_n_wires * f
    return rho, f, theta, j, probs
