import sys

dir_path = "/Users/isidorschoch/Programs/master_thesis/qbc_applications/ipe/"
sys.path.append(dir_path)

import jax
import jax.numpy as jnp
from qbc_ipe_jax_probs import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


num_train_samples = 1024
num_test_samples = num_train_samples // 2**3
func_to_approx = (
    lambda x: jnp.sin(x - jnp.pi / 2.0)
    - 2 * jnp.abs(jnp.sin(1.1 * x - jnp.pi / 2.0)) ** 1.5
)
(
    key,
    W_key,
    b_key,
    x_key_train,
    x_key_test,
    y_key_train,
    y_key_test,
    idx_key,
) = jax.random.split(key, 8)
x_train = jax.random.uniform(
    x_key_train, (1, num_train_samples), minval=0, maxval=1.5 * jnp.pi
)
x_test = jax.random.uniform(
    x_key_test, (1, num_test_samples), minval=0, maxval=1.5 * jnp.pi
)
x_train = jnp.sort(x_train)

y_train = (
    func_to_approx(x_train)
    + jax.random.normal(y_key_train, (1, num_train_samples)) * 0.1
)
y_test = (
    func_to_approx(x_test) + jax.random.normal(y_key_test, (1, num_test_samples)) * 0.1
)
# Initialize random model coefficients

# Initialize random model coefficients
layer_sizes = [1, 16, 16, 4, 1]
num_layers = len(layer_sizes) - 1
quantum_layers = [False for i in range(num_layers)]
quantum_layers[1] = True
quantum_layers[2] = False
weight_dims = list(zip(layer_sizes[1:], layer_sizes[:-1]))
activation_funcs = [jax.nn.tanh for _ in range(num_layers)]
activation_funcs[-1] = lambda x: x

bias_dims = layer_sizes[1:]

weight_list = []
bias_list = []
for i in range(num_layers):
    key, subkey = jax.random.split(key)
    W = jax.random.normal(subkey, weight_dims[i])
    weight_list.append(W)
    b = jax.random.normal(key, (bias_dims[i], 1))
    bias_list.append(b)

qbc_inner = QBCIPEJax(jit_me=True, mode="fwd")


def predict_inner(W_list, b_list, x):
    res = x
    for W, b, f, q_l in zip(W_list, b_list, activation_funcs, quantum_layers):
        tmp = jnp.matmul(W, res) if q_l is False else qbc_inner.matmul(W, res)
        res = f(tmp + b)
    return res


@jax.jit
def loss(W_list, b_list, x, y):
    preds = predict_inner(W_list, b_list, x)
    return 0.5 * jnp.mean((preds - y) ** 2)


@jax.jit
def loss_and_predict(W_list, b_list, x, y):
    preds = predict_inner(W_list, b_list, x)
    return 0.5 * jnp.mean((preds - y) ** 2), preds


lr = 0.05
num_epochs = 201
batch_size = 32

y_preds_begin = predict_inner(weight_list, bias_list, x_train).flatten()

y_preds_list_epochs = []
epochs_list = []
train_loss_list_epochs = [-1, -1]
test_loss_list_epochs = [-1, -1]
epoch_period = 1
for epoch_idx in range(num_epochs):
    if epoch_idx % epoch_period == 0:
        train_loss_val, y_preds = loss_and_predict(
            weight_list, bias_list, x_train, y_train
        )
        test_loss_list_epochs.append(loss(weight_list, bias_list, x_test, y_test))
        print(
            "epoch_idx =",
            epoch_idx,
            ", train loss =",
            train_loss_val,
            ", test loss =",
            test_loss_list_epochs[-1],
        )
        train_loss_list_epochs.append(train_loss_val)
        epochs_list.append(epoch_idx)
        y_preds_list_epochs.append(y_preds.flatten())
    idx_key, subkey = jax.random.split(idx_key)
    rnd_indices = jax.random.randint(idx_key, (batch_size,), 0, x_train.shape[1])
    idx_key = subkey
    W_grad, b_grad = jax.jacfwd(loss, argnums=(0, 1))(
        weight_list, bias_list, x_train[:, rnd_indices], y_train[:, rnd_indices]
    )
    bias_list = [b_l - lr * b_grad_l for b_l, b_grad_l in zip(bias_list, b_grad)]
    weight_list = [W_l - lr * W_grad_l for W_l, W_grad_l in zip(weight_list, W_grad)]

x_train = x_train.flatten()
y_train = y_train.flatten()

data_arr = jnp.array(
    [
        x_train,
        y_train,
        *y_preds_list_epochs,
    ]
)
data_arr = jnp.append(
    jnp.array([train_loss_list_epochs, test_loss_list_epochs]).T, data_arr, axis=1
)


df = pd.DataFrame(data_arr)
df.index = [-2, -1, *epochs_list]
df.to_csv("data.csv", header=True, index=True)
