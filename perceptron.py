import jax
import jax.numpy as jnp
from qbc_ipe_jax_pennylane import *

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


def sigmoid(x):
    return (jnp.tanh(x / 2.0) + 1.0) / 2.0


# @jax.jit
def predict(W, b, inputs):
    res = []
    for x in inputs:
        print("x = ", x)
        z = qbc_ipe_fwd(W, x, 3) + b
        # z = jnp.inner(W, x) + b
        f_z = sigmoid(z)
        res.append(f_z)
    return jnp.array(res)


def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    loss = -jnp.sum(jnp.log(label_probs))
    return loss


b_grad = jax.grad(loss, argnums=1)(W, b)
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
