import pennylane as qml
from pennylane import numpy as np

n_wires = 4
dev = qml.device("default.qubit", wires=n_wires, shots=1)


def binary_to_int(l):
    n = len(l)
    return np.sum([2 ** (n - (i + 1)) for i in range(n) if l[i] == 1])


def add_k_fourier(k, wires):
    for j in range(len(wires)):
        qml.RZ(k * np.pi / (2**j), wires=wires[j])


@qml.qnode(dev)
def addition(m, k):
    qml.BasisEmbedding(m, wires=range(n_wires))  # m encoding
    qml.QFT(wires=range(n_wires))  # step 1
    qml.apply(add_k_fourier(k, range(n_wires)))  # step 2
    qml.adjoint(qml.QFT)(wires=range(n_wires))  # step 3
    return qml.sample()


def subtraction(m, k):
    return addition(-m, k)


def addition_int_result(m, k):
    return binary_to_int(addition(m, k))


def subtraction_int_result(m, k):
    return binary_to_int(subtraction(m, k))
