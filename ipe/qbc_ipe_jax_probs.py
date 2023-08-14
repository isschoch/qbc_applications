import pennylane as qml
import jax.numpy as jnp
import jax
from functools import partial
import numpy as np
from enum import Enum

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)


def qbc_ipe_probs(x, y, num_t_wires=8, num_shots=1, num_n_wires=4):
    assert len(x) == len(y)

    x = x.at[0].set(jnp.finfo(jnp.float32).eps + x[0])
    y = y.at[0].set(jnp.finfo(jnp.float32).eps + y[0])
    M = 2**num_t_wires

    a = 0.5 * (
        1.0 + jnp.inner(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))
    )  # value in (0, 1]
    w = jnp.arcsin(jnp.sqrt(a)) / jnp.pi
    range_vals = jnp.arange(M)
    phase_values = range_vals / M
    closest_integer_idx = jnp.argmin(jax.lax.abs(phase_values - w))
    delta = w - phase_values[closest_integer_idx]

    probs = jnp.sin((M * delta - range_vals) * jnp.pi) ** 2 / (
        M**2 * jnp.sin((delta - range_vals / M) * jnp.pi) ** 2
    )

    j = jnp.argmax(probs) - closest_integer_idx
    # j = jax.random.choice(key, probs, shape=(num_shots,)) - closest_integer_idx

    rho = -(
        (1.0 - 2.0 * jnp.sin(jnp.pi * j / (2**num_t_wires)) ** 2)
        * jnp.linalg.norm(x)
        * jnp.linalg.norm(y)
    )
    return rho


num_t_wires = 12
num_shots = None
partial_qbc_ipe_jax = partial(
    qbc_ipe_probs, num_t_wires=num_t_wires, num_shots=num_shots, num_n_wires=2
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
        return self.switch_dict[self._jit_me, self._mode](x, y)

    def matvec(self, A, x):
        return jax.vmap(self.__call__, in_axes=(0, None))(A, x)

    def matmul(self, A, B):
        return jax.vmap(
            self.matvec,
            in_axes=(None, 1),
            out_axes=1,
        )(A, B)


if __name__ == "__main__":
    A = jnp.array(np.random.rand(4, 4) - 0.5)
    B = jnp.array(np.random.rand(4, 4) - 0.5)
    x = jnp.array(np.random.rand(4) - 0.5)
    y = jnp.array(np.random.rand(4) - 0.5)

    qbc_inner = QBCIPEJax(jit_me=False)
    print(qbc_inner(x, y), jnp.inner(x, y))
    print(qbc_inner.matmul(A, B))
    print(jnp.matmul(A, B))
