import jax
import jax.numpy as jnp


@jax.custom_jvp
@jax.custom_vjp
def log1pexp(x):
    return jnp.logaddexp(0, x)


@log1pexp.defjvp
def log1pexp_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = log1pexp(x)
    tangent_out = x_dot / (1 + jnp.exp(-x))
    return primal_out, tangent_out


def log1pexp_fwd(x):
    y = log1pexp(x)
    return y, y


def log1pexp_bwd(y, g):
    x = jnp.log(jnp.exp(y) - 1)
    return (g * jnp.exp(x) / (jnp.exp(x) + 1),)


log1pexp.defvjp(log1pexp_fwd, log1pexp_bwd)

a = jax.jacfwd(log1pexp)(0.5)
print("a = ", a)
