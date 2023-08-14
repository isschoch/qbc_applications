import pennylane as qml
from pennylane import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy import linalg as la


def qbc_conv(x, y, k, num_t_wires=10):
    assert len(x) == len(y)
    num_n_wires = int(np.ceil(np.log2(len(x))))
    num_c_wires = num_n_wires

    t_wires = range(0, num_t_wires)
    c_wires = range(num_t_wires, num_t_wires + num_n_wires)
    n_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + num_c_wires)
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

    def U_y_alt():
        for one_idx in y_indices:
            qml.ctrl(qml.PauliX, control=n_wires, control_values=one_idx)(
                wires=o_wires[1]
            )

    def qft_adder():
        qml.adjoint(qml.QFT(wires=c_wires))  # step 3
        for n_idx, n_wire in enumerate(n_wires):
            qml.ctrl(add_k_fourier, control=n_wire)(
                2 ** (num_n_wires - n_idx - 1), wires=c_wires
            )
        qml.QFT(wires=c_wires)  # step 1

    def add_k_fourier(l, wires):
        for j in range(len(wires)):
            qml.RZ(l * np.pi / (2**j), wires=wires[j])

    # Copies entries from n register to c register
    def U_minus():
        for i in range(num_n_wires):
            qml.CNOT(wires=[n_wires[i], c_wires[i]])
        for c_wire in c_wires:
            qml.PauliX(wires=c_wire)
        qml.QFT(wires=c_wires)
        add_k_fourier(k + 1, wires=c_wires)
        qml.adjoint(qml.QFT(wires=c_wires))

    # Copies entries from n register to c register
    def U_minus_alt():
        # for i in range(num_n_wires):
        #     qml.CNOT(wires=[n_wires[i], c_wires[i]])
        for n_wire in n_wires:
            qml.PauliX(wires=n_wire)
        qml.QFT(wires=n_wires)
        add_k_fourier(k + 1, wires=n_wires)
        qml.adjoint(qml.QFT(wires=n_wires))

    # Phase oracle used in Grover operator
    def phase_oracle():
        U_x()
        # U_minus()
        U_minus_alt()
        U_y_alt()
        qml.adjoint(U_minus_alt)()
        # U_y()
        qml.CZ(wires=[o_wires[0], o_wires[1]])
        U_minus_alt()
        qml.adjoint(U_y_alt)()
        qml.adjoint(U_minus_alt)()
        # qml.adjoint(U_minus)()
        qml.adjoint(U_x)()

    # Grover operator used in QPE
    def grover_operator():
        phase_oracle()
        qml.GroverOperator(wires=n_wires)

    @qml.qnode(dev)
    def test_fct():
        qml.BasisEmbedding(k, wires=n_wires)
        U_minus()
        U_x()
        U_y()
        return qml.probs(wires=n_wires)

    # QBC circuit
    @qml.qnode(dev)
    def qbc():
        for i in n_wires:
            qml.Hadamard(wires=i)

        my_unitary = qml.matrix(grover_operator)()
        qml.QuantumPhaseEstimation(
            my_unitary,
            target_wires=range(num_t_wires + num_c_wires, num_tot_wires),
            estimation_wires=t_wires,
        )

        return qml.probs(wires=t_wires)

    # return test_fct()
    probs = qbc()
    j = np.argmax(probs)
    theta = 2 * np.pi * j / 2**num_t_wires
    f = np.sin(theta / 2) ** 2
    rho = 2**num_n_wires * f
    return rho, f, theta, j, probs


x = np.random.randint(0, 2, size=(16))
y = np.random.randint(0, 2, size=(16))


print("x =", x.reshape(4, 4))
print("y =", y.reshape(4, 4))


def circular_conv(x, y):
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y))


for k in range(len(x)):
    rho, _, _, _, probs = qbc_conv(x, y, k, 6)

    def convolution_analytic(x, y, k):
        sum = 0
        for i in range(len(x)):
            sum += x[i] * y[(k + 1 - (i + 1) + len(x)) % len(y)]
        return sum

    print(
        "x * y = {analytic}, rho = {rho}".format(
            analytic=convolution_analytic(x, y, k), rho=rho
        )
    )

print("circular_conv(x, y) =", circular_conv(x, y))
