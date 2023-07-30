import jax
import jax.numpy as jnp

from qbc_ipe_jax_pennylane import *

# from qbc_ipe_jax_probs import *
import matplotlib.pyplot as plt
import seaborn as sns


num_samples = 64
func_to_approx = lambda x: jnp.sin(x - jnp.pi / 2.0) + 0.5 * jnp.sin(
    2.0 * x - jnp.pi / 2.0
)
key, W_key, b_key, x_key, y_key, idx_key = jax.random.split(key, 6)
# x_data = jnp.array([[x for x in jnp.linspace(0, 2.0 * jnp.pi, num_samples)]])
x_data = jax.random.uniform(x_key, (1, num_samples), minval=0, maxval=2 * jnp.pi)
print(x_data.shape[1])
print("x_data.shape = ", x_data.shape)
y_data = func_to_approx(x_data) + jax.random.normal(y_key, (1, num_samples)) * 0.05
# Initialize random model coefficients

# Initialize random model coefficients
# key, W_key, b_key = jax.random.split(key, 3)
layer_sizes = [1, 2, 4, 1]
num_layers = len(layer_sizes) - 1
quantum_layers = [False, True, False]
weight_dims = list(zip(layer_sizes[1:], layer_sizes[:-1]))
activation_funcs = [jax.nn.sigmoid for _ in range(num_layers)]
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


def loss(W_list, b_list, x, y):
    preds = predict_inner(W_list, b_list, x)
    return 0.5 * jnp.mean((preds - y) ** 2)


def main():
    global weight_list
    global bias_list
    global x_data
    global y_data
    lr = 0.9
    num_epochs = 100
    batch_size = 8

    y_preds_begin = predict_inner(weight_list, bias_list, x_data).flatten()
    for epoch_idx in range(num_epochs):
        if epoch_idx % 10 == 0:
            print(
                "epoch_idx =",
                epoch_idx,
                "loss =",
                loss(weight_list, bias_list, x_data, y_data),
            )
        # rnd_indices = jnp.array(random.sample(range(x_data.shape[1]), 10))
        rnd_indices = jax.random.randint(idx_key, (batch_size,), 0, x_data.shape[1])
        # for x, y in zip(x_data[rnd_indices], targets[rnd_indices]):
        W_grad, b_grad = jax.jacfwd(loss, argnums=(0, 1))(
            weight_list, bias_list, x_data[:, rnd_indices], y_data[:, rnd_indices]
        )
        bias_list = [b_l - lr * b_grad_l for b_l, b_grad_l in zip(bias_list, b_grad)]
        weight_list = [
            W_l - lr * W_grad_l for W_l, W_grad_l in zip(weight_list, W_grad)
        ]

    y_preds = predict_inner(weight_list, bias_list, x_data).flatten()
    x_data = x_data.flatten()
    y_data = y_data.flatten()

    print("weight_list = ", weight_list)
    print("bias_list = ", bias_list)
    plt.plot(x_data, y_preds_begin, "o", label="predictions_begin")
    plt.plot(x_data, y_preds, "o", label="predictions")
    plt.plot(x_data, y_data, "o", label="data")
    plt.legend(["prediction_begin", "predictions_end", "target"])
    plt.show()


if __name__ == "__main__":
    main()
