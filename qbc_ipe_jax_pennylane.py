import pennylane as qml

# from pennylane import numpy as jnp

import jax.numpy as jnp
import jax
from jax import make_jaxpr
from jax import custom_jvp
import math
from functools import partial
from jax import config

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def qbc_ipe_algorithm(x, y, num_t_wires=8, num_shots=1):
    assert len(x) == len(y)
    num_n_wires = int(jnp.ceil(jnp.log2(len(x))))
    num_tot_wires = num_t_wires + num_n_wires + 1

    t_wires = range(0, num_t_wires)
    n_wires = range(num_t_wires, num_t_wires + num_n_wires)
    a_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + 1)
    tot_wires = range(0, num_t_wires + num_n_wires + 1)

    dev = qml.device("default.qubit.jax", wires=tot_wires, shots=None)

    tmp = jnp.zeros((2**num_n_wires, 2**num_n_wires - 1))

    x = x.at[0].set(jnp.finfo(jnp.float32).eps + x[0])
    A_x = jnp.concatenate((x.reshape(-1, 1), tmp), axis=1)

    Q_x, _ = jnp.linalg.qr(A_x, mode="full")
    for i in range(2**num_n_wires):
        Q_x.at[:, i].set(Q_x[:, i] / jnp.linalg.norm(Q_x[:, i]))

    u_x = Q_x.T
    u_x *= jnp.sign(u_x[0, 0]) * jnp.sign(x[0])

    # Oracle A encoding x data

    def U_x():
        qml.QubitUnitary(u_x, wires=n_wires)

    y = y.at[0].set(jnp.finfo(jnp.float32).eps + y[0])
    A_y = jnp.concatenate((y.reshape(-1, 1), tmp), axis=1)

    Q_y, _ = jnp.linalg.qr(A_y, mode="full")
    for i in range(2**num_n_wires):
        Q_y.at[:, i].set(Q_y[:, i] / jnp.linalg.norm(Q_y[:, i]))

    u_y = Q_y.T
    u_y *= jnp.sign(u_y[0, 0]) * jnp.sign(y[0])

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
            control_values=jnp.ones(num_n_wires),
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

        return qml.probs(wires=t_wires)

    probs = qbc_ipe(U_x, U_y)
    j = jnp.argmax(probs)

    rho = (
        -(1.0 - 2.0 * jnp.sin(jnp.pi * j / (2**num_t_wires)) ** 2)
        * jnp.linalg.norm(x)
        * jnp.linalg.norm(y)
    )
    # print("rho = ", rho, "rho_exact = ", jnp.inner(x, y))
    return rho


# Build a toy dataset.
inputs = jnp.array(
    [
        [0.52, 1.12, 0.77, -1.07],
        [0.88, -1.08, 0.15, 0.26],
        [0.52, 0.06, -1.30, 0.51],
        [0.74, -2.49, 1.39, 0.62],
    ]
)
targets = jnp.array([True, True, False, True])

# Initialize random model coefficients
key, W_key, b_key = jax.random.split(key, 3)
W = jnp.array([-1.6193685, 1.28386154, -1.13687517, -0.4885566])
b = 0.21635686180382116
print("W = ", W)
print("b = ", b)


@qbc_ipe_algorithm.defjvp
def qbc_ipe_algorithm_jvp(num_t_wires, num_shots, primals, tangents):
    primal0, primal1 = primals
    vector0_dot, vector1_dot = tangents
    primal_out = qbc_ipe_algorithm(primal0, primal1)
    tangent_out_1 = qbc_ipe_algorithm(primal1, vector0_dot)
    tangent_out_2 = qbc_ipe_algorithm(primal0, vector1_dot)

    tangent_out = tangent_out_1 + tangent_out_2

    # print(
    #     "primal_out = ", primal_out, "primal_out_exact = ", jnp.inner(primal0, primal1)
    # )
    # print(
    #     "tangent_out = ",
    #     tangent_out,
    #     "tangent_out_exact = ",
    #     jnp.inner(primal1, vector0_dot) + jnp.inner(primal0, vector1_dot),
    # )

    return primal_out, tangent_out


def sigmoid(x):
    return (jnp.tanh(x / 2.0) + 1.0) / 2.0


def predict(W, b, inputs):
    res = []
    for x in inputs:
        print("x = ", x)
        z = qbc_ipe_algorithm(W, x) + b
        # z = jnp.dot(W, x) + b
        f_z = sigmoid(z)
        res.append(f_z)
    return jnp.array(res)


def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    loss = -jnp.sum(jnp.log(label_probs))
    return loss


b_grad = jax.jacrev(loss, argnums=1)(W, b)
W_grad = jax.jacfwd(loss, argnums=0)(W, b)
print("b_grad = ", b_grad)
print("W_grad = ", W_grad)


# W_grad =  [-2.10045795  2.51364099 -2.22189626 -0.26447761], b_grad =  -1.8412136122910145 t = 4
# W_grad =  [-1.08760765  3.00596065 -2.60727465  0.00557337], b_grad =  -1.6791775542536875 t = 5
# W_grad =  [-1.54781114  3.1503103  -2.6269267  -0.14023218], b_grad =  -1.654636884938736 t = 6
# W_grad =  [-1.45221736  3.13848449 -2.74384554 -0.08461535], b_grad =  -1.6611388572331967 t = 7
# W_grad =  [-1.41364956  3.1299076  -2.70705029 -0.06190097], b_grad =  -1.680718223437469 t = 8
# W_grad =  [-1.41313413  3.10491427 -2.69366255 -0.12296308], b_grad =  -1.6581968368063529 t = 9
# W_grad =  [-1.41939689  3.11794621 -2.68710328 -0.11651904], b_grad =  -1.669864790191127 t = 10
# W_grad =  [-1.4293233   3.10943251 -2.68555112 -0.11471314], b_grad =  -1.6696173960894887 t = 11

# W_grad =  [-1.42820196  3.11378908 -2.68983649 -0.11220568], b_grad =  -1.668494062179321 exact
