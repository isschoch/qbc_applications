# import pennylane as qml

# from pennylane import numpy as jnp

import jax.numpy as jnp
import jax
from jax import make_jaxpr
from jax import custom_jvp
import math
from functools import partial
from jax import config

config.update("jax_debug_nans", True)

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)
# from pennylane import numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def qbc_ipe_algorithm(x, y, num_t_wires=8, num_shots=1):
    assert len(x) == len(y)
    num_n_wires = int(jnp.ceil(jnp.log2(len(x))))
    num_tot_wires = num_t_wires + num_n_wires + 1

    # #     return qml.shadow_expval([qml.PauliZ(i) for i in t_wires])
    x = x.at[0].set(jnp.finfo(jnp.float32).eps + x[0])
    y = y.at[0].set(jnp.finfo(jnp.float32).eps + y[0])

    a = 0.5 * (
        1.0 - jnp.inner(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))
    )  # value in (0, 1]
    w = 2.0 * jnp.arcsin(jnp.sqrt(a)) / (jnp.pi)  # value in (0, 1]
    M = 2**num_t_wires

    # # probs = qbc_ipe(U_x, U_y)
    def dist(w0, w1):
        z = set(range(-10, 10))  # change the range as needed
        min_dist = float("inf")

        vals = jnp.array([abs(z_val + w1 - w0) for z_val in z])

        min_dist = jnp.min(vals)

        return min_dist

    distances = jnp.array([dist(w, x / M) for x in range(M)])
    # print("distances =", distances)
    probs_tmp = jnp.array(
        [
            jnp.sin(M * d * jnp.pi) ** 2 / (M**2 * jnp.sin(d * jnp.pi) ** 2)
            for d in distances
        ]
    )
    probs = jnp.where(distances == 0.0, 1.0, probs_tmp)
    # print("probs =", probs)
    # j = jnp.argmax(probs)
    # print("probs =", probs)
    j_values = jnp.arange(2**num_t_wires) * 1.0

    j = jnp.argmax(probs)

    rho = (
        -(1.0 - 2.0 * jnp.sin(jnp.pi * j / (2**num_t_wires)) ** 2)
        * jnp.linalg.norm(x)
        * jnp.linalg.norm(y)
    )
    print("rho = ", rho, "rho_exact = ", jnp.inner(x, y))
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

    print(
        "primal_out = ", primal_out, "primal_out_exact = ", jnp.inner(primal0, primal1)
    )
    print(
        "tangent_out = ",
        tangent_out,
        "tangent_out_exact = ",
        jnp.inner(primal1, vector0_dot) + jnp.inner(primal0, vector1_dot),
    )

    return primal_out, tangent_out


def sigmoid(x):
    return (jnp.tanh(x / 2.0) + 1.0) / 2.0


def predict(W, b, inputs):
    res = []
    for x in inputs:
        print("x = ", x)
        z = qbc_ipe_algorithm(W, x) + b
        f_z = sigmoid(z)
        res.append(f_z)
    return jnp.array(res)


def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))


b_grad = jax.jacfwd(loss, argnums=1)(W, b)
W_grad = jax.jacfwd(loss, argnums=0)(W, b)
print("b_grad = ", b_grad)
print("W_grad = ", W_grad)
