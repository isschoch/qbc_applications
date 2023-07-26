import pennylane as qml
import jax.numpy as jnp
import jax
from functools import partial
import numpy as np

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)


@jax.custom_vjp
def qbc_ipe_rev(x, y, num_t_wires=8, num_shots=None):
    result = qbc_ipe_jax(x, y, num_t_wires, num_shots)
    return result


def qbc_ipe_vjp_fwd(x, y, num_t_wires, num_shots):
    return qbc_ipe_rev(x, y, num_t_wires, num_shots), (y, x)


def qbc_ipe_vjp_bwd(res, g):
    y, x = res
    return (
        g * y,
        g * x,
        None,
        None,
    )  # None return value signifies non-differentiable argument


def qbc_ipe_jax(x, y, num_t_wires=6, num_shots=None):
    assert len(x) == len(y)
    # num_n_wires = int(jnp.ceil(jnp.log2(len(x))))
    num_n_wires = 2
    num_tot_wires = num_t_wires + num_n_wires + 1

    t_wires = range(0, num_t_wires)
    n_wires = range(num_t_wires, num_t_wires + num_n_wires)
    a_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + 1)
    tot_wires = range(0, num_tot_wires)

    dev = qml.device("default.qubit.jax", wires=tot_wires, shots=num_shots)

    tmp = jnp.zeros((2**num_n_wires, 2**num_n_wires - 1))

    x = x.at[0].set(jnp.finfo(jnp.float32).eps + x[0])
    A_x = jnp.concatenate(
        (jnp.pad(x, pad_width=(0, 2**num_n_wires - len(x))).reshape(-1, 1), tmp),
        axis=1,
    )

    Q_x, _ = jnp.linalg.qr(A_x, mode="full")
    for i in range(2**num_n_wires):
        Q_x.at[:, i].set(Q_x[:, i] / jnp.linalg.norm(Q_x[:, i]))

    u_x = Q_x.T
    u_x *= jnp.sign(u_x[0, 0]) * jnp.sign(x[0])

    y = y.at[0].set(jnp.finfo(jnp.float32).eps + y[0])
    A_y = jnp.concatenate(
        (jnp.pad(y, pad_width=(0, 2**num_n_wires - len(y))).reshape(-1, 1), tmp),
        axis=1,
    )

    Q_y, _ = jnp.linalg.qr(A_y, mode="full")
    for i in range(2**num_n_wires):
        Q_y.at[:, i].set(Q_y[:, i] / jnp.linalg.norm(Q_y[:, i]))

    u_y = Q_y.T
    u_y *= jnp.sign(u_y[0, 0]) * jnp.sign(y[0])

    # Phase oracle used in Grover operator
    def A_operator(u_x, u_y):
        qml.Hadamard(wires=a_wires)
        qml.ControlledQubitUnitary(u_x, wires=n_wires, control_wires=a_wires[0])
        qml.PauliX(wires=a_wires)
        qml.ControlledQubitUnitary(u_y, wires=n_wires, control_wires=a_wires[0])
        qml.PauliX(wires=a_wires)
        qml.Hadamard(wires=a_wires)

    # Grover operator used in QPE
    def grover_operator(u_x, u_y):
        # Reflection about "good" state
        qml.PauliZ(wires=a_wires)

        A_operator(u_x, u_y)

        # Reflection about |0> state
        for i in n_wires:
            qml.PauliX(wires=i)
        qml.PauliX(wires=a_wires)
        qml.Hadamard(wires=a_wires)
        qml.MultiControlledX(wires=(*n_wires, *a_wires))
        qml.Hadamard(wires=a_wires)
        for i in n_wires:
            qml.PauliX(wires=i)
        qml.PauliX(wires=a_wires)

        qml.adjoint(A_operator)(u_x, u_y)

    # QBC circuit
    # @jax.jit
    @qml.qnode(dev, interface=None)
    def qbc_ipe(u_x, u_y):
        A_operator(u_x, u_y)

        for t in t_wires:
            qml.Hadamard(wires=t)

        for idx, t in enumerate(t_wires):
            for i in range(2 ** (num_t_wires - idx - 1)):
                qml.ctrl(grover_operator, control=t)(u_x, u_y)

        qml.adjoint(qml.QFT(wires=t_wires))

        return qml.probs(wires=t_wires)

    probs = qbc_ipe(u_x, u_y)
    j = jnp.argmax(probs)

    rho = (
        -(1.0 - 2.0 * jnp.sin(jnp.pi * j / (2**num_t_wires)) ** 2)
        * jnp.linalg.norm(x)
        * jnp.linalg.norm(y)
    )

    return rho


class Test:
    def __init__(self, num_t_wires=2, num_shots=None):
        self._num_t_wires = num_t_wires
        self._num_shots = num_shots
        self._ipe_provider = partial(qbc_ipe_fwd, self._num_t_wires, self._num_shots)


# test = Test(num_t_wires=2, num_shots=None)

qbc_ipe_rev.defvjp(qbc_ipe_vjp_fwd, qbc_ipe_vjp_bwd)


@partial(jax.custom_jvp)
def qbc_ipe_fwd(x, y):
    return qbc_ipe_jax(x, y)


jitted_qbc_ipe_fwd = jax.jit(qbc_ipe_jax)


@qbc_ipe_fwd.defjvp
def qbc_ipe_jvp(primals, tangents):
    primal0, primal1 = primals
    vector0_dot, vector1_dot = tangents
    primal_out = qbc_ipe_fwd(primal0, primal1)

    tangent_out = qbc_ipe_fwd(primal1, vector0_dot) + qbc_ipe_fwd(primal0, vector1_dot)

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


@jax.custom_jvp
def jit_qbc_ipe_fwd(x, y):
    return jitted_qbc_ipe_fwd(x, y)


@jit_qbc_ipe_fwd.defjvp
def jit_qbc_ipe_jvp(primals, tangents):
    primal0, primal1 = primals
    vector0_dot, vector1_dot = tangents
    primal_out = jitted_qbc_ipe_fwd(primal0, primal1)
    tangent_out_1 = jitted_qbc_ipe_fwd(primal1, vector0_dot)
    tangent_out_2 = jitted_qbc_ipe_fwd(primal0, vector1_dot)

    tangent_out = tangent_out_1 + tangent_out_2
    return primal_out, tangent_out


class QBCIPEJax:
    def __init__(
        self, num_n_wires=2, num_t_wires=2, num_shots=None, jit_me=True
    ) -> None:
        self._num_n_wires = num_n_wires
        self._num_t_wires = num_t_wires
        self._num_shots = num_shots
        self._jit_me = jit_me
        self._func = Test(num_t_wires=self._num_t_wires, num_shots=self._num_shots)

    # @jax.custom_vjp

    def __call__(self, x, y):
        return jit_qbc_ipe_fwd(x, y) if self._jit_me is True else qbc_ipe_fwd(x, y)
