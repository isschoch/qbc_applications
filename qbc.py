import pennylane as qml
from pennylane import numpy as np
from itertools import chain
import matplotlib.pyplot as plt

num_t_wires = 2
num_n_wires = 2

t_wires = range(0, num_t_wires)
n_wires = range(num_t_wires, num_t_wires + num_n_wires)
o_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + 2)
tot_wires = range(0,  num_t_wires + num_n_wires + 2)

dev = qml.device("default.qubit", wires=tot_wires, shots=1)

def U_x():
    # oracle A
    # for i in n_wires:
    #     qml.CNOT(wires=[i, o_wires[0]])    
    # qml.PauliX(wires=o_wires[0])
    qml.PauliX(wires=o_wires[0])

def U_y():
    # oracle B
    for i in n_wires:
        qml.CNOT(wires=[i, o_wires[1]])
    # qml.PauliX(wires=o_wires[1])

def phase_oracle(U_x, U_y):
    U_x()
    U_y()
    qml.CZ(wires=[o_wires[0], o_wires[1]])
    U_y()
    U_x()

def grover_operator(U_x, U_y):
    qml.GroverOperator(wires=n_wires)
    phase_oracle(U_x, U_y)

@qml.qnode(dev)
def qbc(U_x, U_y):
    for i in n_wires:
        qml.Hadamard(wires=i)
    my_unitary = qml.matrix(grover_operator)(U_x, U_y)
    qml.QuantumPhaseEstimation(my_unitary, target_wires=range(num_t_wires, num_t_wires + num_n_wires + 2), estimation_wires=t_wires)
    return qml.probs(wires=t_wires)

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
print("t_probs =", probs)
# plt.bar(range(2**num_t_wires), probs)
# plt.show()

j = np.argmax(probs)
print("j =", j)
theta = 2 * np.pi * j / 2**num_t_wires
print("theta =", theta)
f = np.sin(theta/2)**2
print("f =", f)
rho = 2**num_n_wires * f
print("rho =", rho)

x = np.array(U_x_vec_full())
y = np.array(U_y_vec_full())

print("U_x_vec_full =", x)
print("U_y_vec_full =", y)
print("x * y = ", np.inner(x, y))
