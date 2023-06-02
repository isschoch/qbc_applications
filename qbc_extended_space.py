import pennylane as qml
from pennylane import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy import linalg as la


def qbc_conv_algorithm_extended_space(x, y, num_t_wires=10):
    assert len(x) == len(y)
    num_n_wires = int(np.ceil(np.log2(len(x))))
    num_c_wires = num_n_wires

    t_wires = range(0, num_t_wires)
    n_wires = range(num_t_wires, num_t_wires + num_n_wires)
    c_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + num_c_wires)
    o_wires = range(
        num_t_wires + num_n_wires + num_c_wires,
        num_t_wires + num_n_wires + num_c_wires + 2,
    )
    tot_wires = range(0, num_t_wires + num_n_wires + num_c_wires + 2)
    num_tot_wires = len(tot_wires)

    dev = qml.device("default.qubit", wires=tot_wires)

    x_indices = [
        [int(k) for k in format(elem, "0%sb" % num_n_wires)]
        for elem in np.nonzero(x)[0]
    ]

    y_indices = [
        [int(k) for k in format(elem, "0%sb" % num_n_wires)]
        for elem in np.nonzero(y)[0]
    ]

    # Oracle A encoding x data
    def U_x():
        for one_idx in x_indices:
            qml.ctrl(qml.PauliX, control=n_wires, control_values=one_idx)(
                wires=o_wires[0]
            )

    # Oracle B encoding y data
    def U_y():
        for one_idx in y_indices:
            qml.ctrl(qml.PauliX, control=c_wires, control_values=one_idx)(
                wires=o_wires[1]
            )

    # Copies entries from n register to c register
    def U_copy():
        for i in range(num_n_wires):
            qml.CNOT(wires=[n_wires[i], c_wires[i]])

    # Phase oracle used in Grover operator
    def phase_oracle():
        U_x()
        U_copy()
        U_y()
        qml.CZ(wires=[o_wires[0], o_wires[1]])
        qml.adjoint(U_y)()
        qml.adjoint(U_copy)()
        qml.adjoint(U_x)()

    # Grover operator used in QPE
    def grover_operator():
        phase_oracle()
        qml.GroverOperator(wires=n_wires)

    @qml.qnode(dev)
    def test_fct():
        qml.BasisEmbedding(k, wires=n_wires)
        U_copy()
        U_x()
        U_y()
        return qml.probs(wires=n_wires)

    # QBC circuit
    @qml.qnode(dev)
    def qbc():
        for i in n_wires:
            qml.Hadamard(wires=i)

        my_unitary = qml.matrix(grover_operator)()
        print("my_unitary.size = ", np.shape(my_unitary))
        qml.QuantumPhaseEstimation(
            my_unitary,
            target_wires=range(num_t_wires, num_tot_wires),
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

    # return test_fct()
    probs = qbc()
    j = np.argmax(probs)
    theta = 2 * np.pi * j / 2**num_t_wires
    f = np.sin(theta / 2) ** 2
    rho = 2**num_n_wires * f
    return rho, f, theta, j, probs


# x = [1, 1, 0, 0]
# y = [1, 0, 0, 1]

# probs, _, _, _, _ = qbc_conv_algorithm(x, y, 10)

# print(
#     "np.argmax(probs) = {argmax}, probs = {probs}".format(
#         argmax=np.argmax(probs), probs=probs
#     )
# )

# for k in range(len(x)):
#     rho, _, _, _, probs = qbc_conv_algorithm(x, y, 10)

#     print(
#         "x * y = {analytic}, rho = {rho}, probs = {probs}".format(
#             analytic=np.inner(x, y), rho=rho, probs=probs
#         )
#     )
