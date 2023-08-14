import jax
import jax.numpy as jnp
from functools import partial


key = jax.random.PRNGKey(0)
# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12, 0.77, -1.07]])
inputs = jnp.array(
    [
        [0.52, 1.12, 0.77, -1.07],
        [0.88, -1.08, 0.15, 0.26],
        [0.52, 0.06, -1.30, 0.51],
        [0.74, -2.49, 1.39, 0.62],
    ]
)
targets = jnp.array([0.5, 1.0, -1.5, 2.3])

# Initialize random model coefficients
key, W_key, b_key = jax.random.split(key, 3)
W = jax.random.normal(W_key, (4,))
b = jax.random.normal(b_key, ())
print("W = ", W)
print("b = ", b)


@jax.custom_jvp
def qbc_ipe_algorithm(x, y):
    x_norm = jnp.linalg.norm(x)
    y_norm = jnp.linalg.norm(y)
    normalized_inner_prod = jnp.dot(x, y) / (x_norm * y_norm)
    inner_prod = normalized_inner_prod * x_norm * y_norm
    return inner_prod


def just_a_wrapper(x, y):
    return jnp.dot(y, y)


@qbc_ipe_algorithm.defjvp
def qbc_ipe_algorithm_jvp(primals, tangents):
    vector0_dot, vector1_dot = tangents
    primal0, primal1 = primals
    primal_out = just_a_wrapper(primal0, primal1)
    tangent_out_1 = just_a_wrapper(primal1, vector0_dot)
    tangent_out_2 = just_a_wrapper(primal0, vector1_dot)

    tangent_out = tangent_out_1 + tangent_out_2

    return primal_out, tangent_out


def sigmoid(x):
    return (jnp.tanh(x / 2.0) + 1.0) / 2.0


def predict(W, b, inputs):
    res = []
    for x in inputs:
        print("x = ", x)
        z = qbc_ipe_algorithm(W, x) + b
        f_z = sigmoid(z)
        res.append(z)
    return jnp.array(res)


def loss(W, b):
    preds = predict(W, b, inputs)
    # label_probs = preds * targets + (1 - preds) * (1 - targets)
    return jnp.linalg.norm(preds - targets)
    # return -jnp.sum(jnp.log(label_probs))


b_grad = jax.jacfwd(loss, argnums=1)(W, b)
print("b_grad = ", b_grad)
W_grad = jax.jacfwd(loss, argnums=0)(W, b)
print("W_grad = ", W_grad)
