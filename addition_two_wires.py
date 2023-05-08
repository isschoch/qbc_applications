import pennylane as qml
from pennylane import numpy as np

num_n_wires = 4
num_c_wires = 4
n_wires = range(num_n_wires)
c_wires = range(num_n_wires, num_n_wires + num_c_wires)
tot_wires = range(num_n_wires + num_c_wires)

dev = qml.device("default.qubit", wires= tot_wires, shots=1)

def binary_to_int(l):
    n = len(l)
    return np.sum([2**(n-(i+1)) for i in range(n) if l[i] == 1])

def add_k_fourier(k, wires):
    for j in range(len(wires)):
        qml.RZ(k * np.pi / (2**j), wires=wires[j])

@qml.qnode(dev)
def addition_two_wires(m, k):
    qml.BasisEmbedding(m, wires=n_wires) # m encoding
    qml.BasisEmbedding(k, wires=c_wires) # k encoding
    qml.adjoint(qml.QFT)(wires=c_wires) # step 3
    for i in n_wires:
        qml.ctrl(add_k_fourier, control=i)(2**(num_n_wires - i - 1), wires=c_wires)
    qml.QFT(wires=c_wires) # step 1
    # qml.QFT(wires=c_wires) # step 3
    return qml.sample(wires=c_wires)


def subtraction_two_wires(m, k):
    return addition_two_wires(m, k)

def addition_two_wires_int_result(m, k):
    return binary_to_int(addition_two_wires(m, k))

def subtraction_two_wires_int_result(m, k):
    return binary_to_int(subtraction_two_wires(m, k))

print(addition_two_wires(3, 10))
