import pennylane as qml
from pennylane import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def qbc_ipe_algorithm(x, y, num_t_wires=10, num_shots=1):
    assert len(x) == len(y)
    num_n_wires = int(np.ceil(np.log2(len(x))))
    num_tot_wires = num_t_wires + num_n_wires + 1

    t_wires = range(0, num_t_wires)
    n_wires = range(num_t_wires, num_t_wires + num_n_wires)
    a_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + 1)
    tot_wires = range(0, num_t_wires + num_n_wires + 1)

    dev = qml.device("default.qubit", wires=tot_wires, shots=num_shots)

    A_x = np.zeros((2**num_n_wires, 2**num_n_wires))
    A_x[:, 0] = np.pad(np.array(x), pad_width=(0, 2**num_n_wires - len(x)))
    Q_x, _ = la.qr(A_x, mode="full")
    for i in range(2**num_n_wires):
        Q_x[:, i] /= la.norm(Q_x[:, i])

    u_x = Q_x.T
    u_x *= np.sign(u_x[0, 0]) * np.sign(x[0])

    # Oracle A encoding x data
    def U_x():
        # comm_ctr += 1
        qml.QubitUnitary(u_x, wires=n_wires)

    A_y = np.zeros((2**num_n_wires, 2**num_n_wires))
    A_y[:, 0] = np.pad(np.array(y), pad_width=(0, 2**num_n_wires - len(y)))
    Q_y, _ = la.qr(A_y, mode="full")
    for i in range(2**num_n_wires):
        Q_y[:, i] /= la.norm(Q_y[:, i])

    u_y = Q_y.T
    u_y *= np.sign(u_y[0, 0]) * np.sign(y[0])

    # Oracle B encoding y data
    def U_y():
        qml.QubitUnitary(u_y, wires=n_wires)

    # Phase oracle used in Grover operator
    def A_operator(U_x, U_y):
        qml.Hadamard(wires=a_wires)
        qml.ctrl(U_x, control=a_wires, control_values=[0])()
        qml.PauliX(wires=a_wires)
        qml.ctrl(U_y, control=a_wires, control_values=[0])()
        qml.PauliX(wires=a_wires)
        qml.Hadamard(wires=a_wires)

    # Grover operator used in QPE
    def grover_operator():
        # Reflection about "good" state
        qml.PauliZ(wires=a_wires)

        A_operator(U_x, U_y)

        # Reflection about |0> state
        for i in n_wires:
            qml.PauliX(wires=i)
        qml.PauliX(wires=a_wires)
        qml.ctrl(
            qml.PauliZ,
            control=n_wires,
            control_values=np.ones(num_n_wires),
        )(wires=a_wires)
        for i in n_wires:
            qml.PauliX(wires=i)
        qml.PauliX(wires=a_wires)

        qml.adjoint(A_operator)(U_x, U_y)

    # QBC circuit
    @qml.qnode(dev, interface=None)
    def qbc_ipe(U_x, U_y):
        A_operator(U_x, U_y)

        for t in t_wires:
            qml.Hadamard(wires=t)

        for idx, t in enumerate(t_wires):
            for i in range(2 ** (num_t_wires - idx - 1)):
                qml.ctrl(grover_operator, control=t)()

        qml.adjoint(qml.QFT(wires=t_wires))

        return qml.probs(wires=t_wires), qml.counts(wires=t_wires)

    probs, counts = qbc_ipe(U_x, U_y)
    j = np.argmax(probs)
    theta = 2 * np.pi * j / 2**num_t_wires
    f = np.sin(theta / 2) ** 2
    rho = -(1.0 - 2.0 * f) * la.norm(x) * la.norm(y)
    return rho, f, theta, j, probs, counts
