import pennylane as qml
from pennylane import numpy as np
from qbc_applications.addition import *

n_wires = range(0, 4)
o_wires = range(4, 5)
tot_wires = range(0, 5)

dev = qml.device("default.qubit", wires=tot_wires, shots=1)

x = np.random.randint(0, 2, size=(2**4))


def A_oracle(i):
    qml.BasisEmbedding(x[i], wires=o_wires)


@qml.qnode(dev)
def tmp_circuit(i):
    qml.BasisEmbedding(i, wires=n_wires)
    return qml.sample(wires=n_wires)


@qml.qnode(dev)
def circuit():
    i_idx = tmp_circuit(12)
    qml.BasisEmbedding(i_idx, wires=n_wires)
    A_oracle(binary_to_int(i_idx))
    return qml.sample()


def prep_state(i):
    qml.BasisEmbedding(i, wires=n_wires)


@qml.qnode(dev)
def tmp_x_oracle():
    prep_state(3)
    for i in n_wires:
        qml.CNOT(wires=[i, o_wires[0]])

    return qml.sample()


# print("x = ", x)

print(tmp_x_oracle())
