import jax
import jax.numpy as jnp


class Sigmoid:
    def __init__(self, alpha):
        self.alpha = alpha

    @jax.custom_jvp
    def __call__(self, x):
        return 1 / (1 + jnp.exp(-self.alpha * x))

    @__call__.defjvp
    def sigmoid_jvp(primals, tangents):
        self, x = primals
        _, x_dot = tangents
        primal_out = self(x)
        tangent_out = self.alpha * primal_out * (1 - primal_out) * x_dot
        return primal_out, tangent_out


sigmoid = Sigmoid(2.0)
print("sigmoid(0.5) = ", sigmoid(0.5))

# print("jax.grad(sigmoid)(0.5) = ", jax.grad(sigmoid)(0.5))
