# import pennylane as qml

# from pennylane import numpy as jnp

# import jax.numpy as jnp
# import jax
# from jax import make_jaxpr
# from jax import custom_jvp
# import math
# from functools import partial
# from jax import config

# config.update("jax_debug_nans", True)

# jax.config.update("jax_enable_x64", True)
# key = jax.random.PRNGKey(0)
# # from pennylane import numpy as jnp


# @partial(jax.custom_jvp, nondiff_argnums=(2, 3))
# def qbc_ipe_algorithm(x, y, num_t_wires=8, num_shots=1):
#     assert len(x) == len(y)
#     num_n_wires = int(jnp.ceil(jnp.log2(len(x))))
#     num_tot_wires = num_t_wires + num_n_wires + 1

#     # #     return qml.shadow_expval([qml.PauliZ(i) for i in t_wires])
#     x = x.at[0].set(jnp.finfo(jnp.float32).eps + x[0])
#     y = y.at[0].set(jnp.finfo(jnp.float32).eps + y[0])

#     a = 0.5 * (
#         1.0 - jnp.inner(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))
#     )  # value in (0, 1]
#     w = 2.0 * jnp.arcsin(jnp.sqrt(a)) / (jnp.pi)  # value in (0, 1]
#     M = 2**num_t_wires

#     # # probs = qbc_ipe(U_x, U_y)
#     def dist(w0, w1):
#         z = set(range(-10, 10))  # change the range as needed
#         min_dist = float("inf")

#         vals = jnp.array([abs(z_val + w1 - w0) for z_val in z])

#         min_dist = jnp.min(vals)

#         return min_dist

#     distances = jnp.array([dist(w, x / M) for x in range(M)])
#     # print("distances =", distances)
#     probs_tmp = jnp.array(
#         [
#             jnp.sin(M * d * jnp.pi) ** 2 / (M**2 * jnp.sin(d * jnp.pi) ** 2)
#             for d in distances
#         ]
#     )
#     probs = jnp.where(distances == 0.0, 1.0, probs_tmp)
#     # print("probs =", probs)
#     # j = jnp.argmax(probs)
#     # print("probs =", probs)
#     j_values = jnp.arange(2**num_t_wires) * 1.0

#     j = jnp.argmax(probs)

#     rho = (
#         -(1.0 - 2.0 * jnp.sin(jnp.pi * j / (2**num_t_wires)) ** 2)
#         * jnp.linalg.norm(x)
#         * jnp.linalg.norm(y)
#     )
#     print("rho = ", rho, "rho_exact = ", jnp.inner(x, y))
#     return rho


# # Build a toy dataset.
# inputs = jnp.array(
#     [
#         [0.52, 1.12, 0.77, -1.07],
#         [0.88, -1.08, 0.15, 0.26],
#         [0.52, 0.06, -1.30, 0.51],
#         [0.74, -2.49, 1.39, 0.62],
#     ]
# )
# targets = jnp.array([True, True, False, True])

# # Initialize random model coefficients
# key, W_key, b_key = jax.random.split(key, 3)
# W = jnp.array([-1.6193685, 1.28386154, -1.13687517, -0.4885566])
# b = 0.21635686180382116
# print("W = ", W)
# print("b = ", b)


# @qbc_ipe_algorithm.defjvp
# def qbc_ipe_algorithm_jvp(num_t_wires, num_shots, primals, tangents):
#     primal0, primal1 = primals
#     vector0_dot, vector1_dot = tangents
#     primal_out = qbc_ipe_algorithm(primal0, primal1)
#     tangent_out_1 = qbc_ipe_algorithm(primal1, vector0_dot)
#     tangent_out_2 = qbc_ipe_algorithm(primal0, vector1_dot)

#     tangent_out = tangent_out_1 + tangent_out_2

#     print(
#         "primal_out = ", primal_out, "primal_out_exact = ", jnp.inner(primal0, primal1)
#     )
#     print(
#         "tangent_out = ",
#         tangent_out,
#         "tangent_out_exact = ",
#         jnp.inner(primal1, vector0_dot) + jnp.inner(primal0, vector1_dot),
#     )

#     return primal_out, tangent_out


# def sigmoid(x):
#     return (jnp.tanh(x / 2.0) + 1.0) / 2.0


# def predict(W, b, inputs):
#     res = []
#     for x in inputs:
#         print("x = ", x)
#         z = qbc_ipe_algorithm(W, x) + b
#         f_z = sigmoid(z)
#         res.append(f_z)
#     return jnp.array(res)


# def loss(W, b):
#     preds = predict(W, b, inputs)
#     label_probs = preds * targets + (1 - preds) * (1 - targets)
#     return -jnp.sum(jnp.log(label_probs))


# b_grad = jax.jacfwd(loss, argnums=1)(W, b)
# W_grad = jax.jacfwd(loss, argnums=0)(W, b)
# print("b_grad = ", b_grad)
# print("W_grad = ", W_grad)
import pennylane as qml
import jax.numpy as jnp
import jax
from functools import partial
import numpy as np
from enum import Enum

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)


def qbc_ipe_jax(x, y, num_t_wires=8, num_shots=1, num_n_wires=4):
    assert len(x) == len(y)
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
    return rho


# def qbc_ipe_jax(x, y, num_t_wires=8, num_shots=None, num_n_wires=4):
#     assert len(x) == len(y)
#     num_n_wires = num_n_wires
#     num_tot_wires = num_t_wires + num_n_wires + 1

#     t_wires = range(0, num_t_wires)
#     n_wires = range(num_t_wires, num_t_wires + num_n_wires)
#     a_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + 1)
#     tot_wires = range(0, num_tot_wires)

#     dev = qml.device("default.qubit.jax", wires=tot_wires, shots=num_shots)

#     tmp = jnp.zeros((2**num_n_wires, 2**num_n_wires - 1))

#     x = x.at[0].set(jnp.finfo(jnp.float32).eps + x[0])
#     A_x = jnp.concatenate(
#         (jnp.pad(x, pad_width=(0, 2**num_n_wires - len(x))).reshape(-1, 1), tmp),
#         axis=1,
#     )

#     Q_x, _ = jnp.linalg.qr(A_x, mode="full")
#     for i in range(2**num_n_wires):
#         Q_x.at[:, i].set(Q_x[:, i] / jnp.linalg.norm(Q_x[:, i]))

#     u_x = Q_x.T
#     u_x *= jnp.sign(u_x[0, 0]) * jnp.sign(x[0])

#     y = y.at[0].set(jnp.finfo(jnp.float32).eps + y[0])
#     A_y = jnp.concatenate(
#         (jnp.pad(y, pad_width=(0, 2**num_n_wires - len(y))).reshape(-1, 1), tmp),
#         axis=1,
#     )

#     Q_y, _ = jnp.linalg.qr(A_y, mode="full")
#     for i in range(2**num_n_wires):
#         Q_y.at[:, i].set(Q_y[:, i] / jnp.linalg.norm(Q_y[:, i]))

#     u_y = Q_y.T
#     u_y *= jnp.sign(u_y[0, 0]) * jnp.sign(y[0])

#     # Phase oracle used in Grover operator
#     def A_operator(u_x, u_y):
#         qml.Hadamard(wires=a_wires)
#         qml.ControlledQubitUnitary(u_x, wires=n_wires, control_wires=a_wires[0])
#         qml.PauliX(wires=a_wires)
#         qml.ControlledQubitUnitary(u_y, wires=n_wires, control_wires=a_wires[0])
#         qml.PauliX(wires=a_wires)
#         qml.Hadamard(wires=a_wires)

#     # Grover operator used in QPE
#     def grover_operator(u_x, u_y):
#         # Reflection about "good" state
#         qml.PauliZ(wires=a_wires)

#         A_operator(u_x, u_y)

#         # Reflection about |0> state
#         for i in n_wires:
#             qml.PauliX(wires=i)
#         qml.PauliX(wires=a_wires)
#         qml.Hadamard(wires=a_wires)
#         qml.MultiControlledX(wires=(*n_wires, *a_wires))
#         qml.Hadamard(wires=a_wires)
#         for i in n_wires:
#             qml.PauliX(wires=i)
#         qml.PauliX(wires=a_wires)

#         qml.adjoint(A_operator)(u_x, u_y)

#     # QBC circuit
#     @qml.qnode(dev, interface=None)
#     def qbc_ipe(u_x, u_y):
#         A_operator(u_x, u_y)

#         for t in t_wires:
#             qml.Hadamard(wires=t)

#         for idx, t in enumerate(t_wires):
#             for i in range(2 ** (num_t_wires - idx - 1)):
#                 qml.ctrl(grover_operator, control=t)(u_x, u_y)

#         qml.adjoint(qml.QFT(wires=t_wires))

#         return qml.probs(wires=t_wires)

#     probs = qbc_ipe(u_x, u_y)
#     j = jnp.argmax(probs)

#     rho = (
#         -(1.0 - 2.0 * jnp.sin(jnp.pi * j / (2**num_t_wires)) ** 2)
#         * jnp.linalg.norm(x)
#         * jnp.linalg.norm(y)
#     )

#     return rho


num_n_wires = 2
num_t_wires = 7
num_shots = None
partial_qbc_ipe_jax = partial(
    qbc_ipe_jax, num_t_wires=num_t_wires, num_shots=num_shots, num_n_wires=num_n_wires
)
jitted_qbc_ipe_jax = jax.jit(partial_qbc_ipe_jax)


@jax.custom_vjp
def qbc_ipe_rev(x, y):
    return partial_qbc_ipe_jax(x, y)


def qbc_ipe_vjp_fwd(x, y):
    return qbc_ipe_rev(x, y), (y, x)


def qbc_ipe_vjp_bwd(res, g):
    y, x = res
    return (g * y, g * x)


qbc_ipe_rev.defvjp(qbc_ipe_vjp_fwd, qbc_ipe_vjp_bwd)


@jax.custom_vjp
def jit_qbc_ipe_rev(x, y):
    result = jitted_qbc_ipe_jax(x, y)
    # print("jit_qbc_ipe_rev =", result, "result_exact =", jnp.inner(x, y))
    return result


def jit_qbc_ipe_vjp_fwd(x, y):
    return jit_qbc_ipe_rev(x, y), (y, x)


def jit_qbc_ipe_vjp_bwd(res, g):
    y, x = res
    return (g * y, g * x)


jit_qbc_ipe_rev.defvjp(jit_qbc_ipe_vjp_fwd, jit_qbc_ipe_vjp_bwd)


@jax.custom_jvp
def qbc_ipe_fwd(x, y):
    result = partial_qbc_ipe_jax(x, y)
    print("qbc_ipe_fwd =", result, "result_exact =", jnp.inner(x, y))
    return result


@qbc_ipe_fwd.defjvp
def qbc_ipe_jvp(primals, tangents):
    primal0, primal1 = primals
    vector0_dot, vector1_dot = tangents
    primal_out = qbc_ipe_fwd(primal0, primal1)
    tangent_out = qbc_ipe_fwd(primal1, vector0_dot) + qbc_ipe_fwd(primal0, vector1_dot)
    return primal_out, tangent_out


@jax.custom_jvp
def jit_qbc_ipe_fwd(x, y):
    result = jitted_qbc_ipe_jax(x, y)
    print("jit_qbc_ipe_fwd =", result, "result_exact =", jnp.inner(x, y))
    return result


@jit_qbc_ipe_fwd.defjvp
def jit_qbc_ipe_jvp(primals, tangents):
    primal0, primal1 = primals
    vector0_dot, vector1_dot = tangents
    primal_out = jit_qbc_ipe_fwd(primal0, primal1)
    tangent_out = jit_qbc_ipe_fwd(primal1, vector0_dot) + jit_qbc_ipe_fwd(
        primal0, vector1_dot
    )
    return primal_out, tangent_out


class QBCIPEJax:
    def __init__(self, jit_me=True, mode="fwd") -> None:
        self._jit_me = jit_me
        self._mode = mode
        if mode != "fwd" and mode != "rev":
            raise ValueError("mode must be fwd or rev")

        self.switch_dict = {
            (False, "fwd"): qbc_ipe_fwd,
            (True, "fwd"): jit_qbc_ipe_fwd,
            (False, "rev"): qbc_ipe_rev,
            (True, "rev"): jit_qbc_ipe_rev,
        }

    def __call__(self, x, y):
        print("in __call__")
        return self.switch_dict[self._jit_me, self._mode](x, y)

    def matvec(self, A, x):
        return jax.vmap(self.__call__, in_axes=(0, None))(A, x)

    def matmul(self, A, B):
        # return jax.vmap(
        #     self.matvec,
        #     in_axes=(None, 1),
        #     out_axes=1,
        # )(A, B)
        m = A.shape[0]
        k = A.shape[1]
        n = B.shape[1]
        result = []
        for i in range(m):
            for j in range(n):
                result.append(self.__call__(A[i, :], B[:, j]))
        return jnp.array(result).reshape(m, n)


if __name__ == "__main__":
    A = jnp.array(np.random.rand(4, 4) - 0.5)
    B = jnp.array(np.random.rand(4, 4) - 0.5)
    x = jnp.array(np.random.rand(4) - 0.5)

    qbc_inner = QBCIPEJax(jit_me=True)
    print(qbc_inner.matmul(A, B))
    print(jnp.matmul(A, B))
