import pennylane as qml
from pennylane import numpy as np

n_wires = range(0, 4)
o_wires = range(4, 6)
tot_wires = range(0, 6)

dev = qml.device("default.qubit", wires=tot_wires, shots=1)

x = np.random.randint(0, 2, size=(2**4, 1))


@qml.qnode(dev)
def grover_operator():
    phase_oracle()
    qml.GroverOperator(wires=n_wires)

    return qml.state()


def phase_oracle():
    # oracle A
    for i in n_wires:
        qml.CNOT(wires=[i, o_wires[0]])

    # oracle B
    for i in n_wires:
        qml.CNOT(wires=[i, o_wires[1]])

    qml.CZ(wires=[o_wires[0], o_wires[1]])

    # oracle A
    for i in n_wires:
        qml.CNOT(wires=[i, o_wires[0]])

    # oracle B
    for i in n_wires:
        qml.CNOT(wires=[i, o_wires[1]])


print(grover_operator())
