import jax
import jax.numpy as jnp
from qbc_ipe_jax_probs import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py

# import plotly.express as px


num_samples = 1024
func_to_approx = (
    lambda x: jnp.sin(x - jnp.pi / 2.0)
    + 1.5 * jnp.abs(jnp.sin(2.0 * x - jnp.pi / 2.0)) ** 1.5
)
key, W_key, b_key, x_key, y_key, idx_key = jax.random.split(key, 6)
x_data = jax.random.uniform(x_key, (1, num_samples), minval=0, maxval=1.5 * jnp.pi)
x_data = jnp.sort(x_data)
print(x_data.shape[1])
print("x_data.shape = ", x_data.shape)
y_data = func_to_approx(x_data) + jax.random.normal(y_key, (1, num_samples)) * 0.1
# Initialize random model coefficients

# Initialize random model coefficients
layer_sizes = [1, 16, 16, 1]
num_layers = len(layer_sizes) - 1
quantum_layers = [False for i in range(num_layers)]
quantum_layers[1] = False
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


lr = 0.1
num_epochs = 501
batch_size = 32

y_preds_begin = predict_inner(weight_list, bias_list, x_data).flatten()

y_preds_list_epochs = []
weight_list_epochs = []
bias_list_epochs = []
epochs_list = []
loss_list_epochs = []
epoch_period = 50
for epoch_idx in range(num_epochs):
    if epoch_idx % epoch_period == 0:
        loss_val, y_preds = loss_and_predict(weight_list, bias_list, x_data, y_data)
        print("epoch_idx =", epoch_idx, "loss =", loss_val)
        loss_list_epochs.append(loss_val)
        epochs_list.append(epoch_idx)
        y_preds_list_epochs.append(y_preds.flatten())
        weight_list_epochs.append(weight_list)
        bias_list_epochs.append(bias_list)
    idx_key, subkey = jax.random.split(idx_key)
    rnd_indices = jax.random.randint(idx_key, (batch_size,), 0, x_data.shape[1])
    idx_key = subkey
    W_grad, b_grad = jax.jacfwd(loss, argnums=(0, 1))(
        weight_list, bias_list, x_data[:, rnd_indices], y_data[:, rnd_indices]
    )
    bias_list = [b_l - lr * b_grad_l for b_l, b_grad_l in zip(bias_list, b_grad)]
    weight_list = [W_l - lr * W_grad_l for W_l, W_grad_l in zip(weight_list, W_grad)]

# y_preds = y_preds_list_epochs[-1]
x_data = x_data.flatten()
y_data = y_data.flatten()

ax = sns.scatterplot(
    x=x_data,
    y=y_data,
    marker=".",
    linewidth=0,
    alpha=0.4,
    color="black",
    label="data",
)
ax.set_xticks([0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi])
ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"])
ax.set_xlabel("x")
ax.set_ylabel("y")
sns.lineplot(
    x=x_data,
    y=y_preds_list_epochs[0],
    label="initial n.n.",
    linestyle="dotted",
    linewidth=4,
)


data_dict = dict(
    zip(
        [
            "epochs_list",
            "weight_list_epochs",
            "bias_list_epochs",
            "y_preds_list_epochs",
        ],
        [
            epochs_list,
            weight_list_epochs,
            bias_list_epochs,
            y_preds_list_epochs,
        ],
    )
)

import json

json.dump(data_dict, open("data.txt", "w"))
# pd.DataFrame(x_data).to_csv("x_data.csv")


# for y_preds, epoch_idx in zip(y_preds_list_epochs, epochs_list):
#     sns.lineplot(
#         x=x_data,
#         y=y_preds,
#         # label="trained n.n.",
#         linestyle="dashed",
#         linewidth=4,
#         # color="red",
#         hue=epoch_idx / num_epochs,
#         palette="Set1",
#     )

# sns.lineplot(
#     x=x_data,
#     y=y_preds_list_epochs[-1],
#     label="trained n.n.",
#     linestyle="dashed",
#     linewidth=4,
#     color="red",
# )

# plt.show()


# import plotly.express as px
# import plotly.graph_objects as go

# # create a figure with plotly express
# fig = px.scatter(
#     x=x_data,
#     y=y_data,
#     opacity=0.4,
#     color_discrete_sequence=["black"],
#     labels={"x": "x", "y": "y"},
# )

# # add the initial and trained n.n. lines with plotly graph objects
# fig.add_trace(
#     go.Scatter(
#         x=x_data,
#         y=y_preds_list_epochs[0],
#         mode="lines",
#         line=dict(color="blue", dash="dot", width=4),
#         name="initial n.n.",
#         # set visibility to True by default
#         visible=True,
#     ),
# )

# # add traces for each epoch in y_preds_list_epochs
# for epoch in epochs_list:
#     fig.add_trace(
#         go.Scatter(
#             x=x_data,
#             y=y_preds_list_epochs[epoch // epoch_period],
#             mode="lines",
#             line=dict(color="red", width=4),
#             name=f"Epoch {epoch:4d}, loss={(loss_list_epochs[epoch // epoch_period]):.3f}",
#             # set visibility to False by default
#             visible=False,
#         )
#     )

# # create a list of steps for the slider
# steps = []
# for epoch in epochs_list:
#     step = dict(
#         method="update",
#         args=[
#             {"visible": [True, True] + [False] * (len(fig.data) - 2)}
#         ],  # set all traces except the first two invisible
#         label=f"{epoch}",  # set label for slider step
#     )
#     # set the (i + 2)th trace (corresponding to the ith epoch) visible
#     step["args"][0]["visible"][epoch // epoch_period + 2] = True
#     steps.append(step)

# # create a slider object with the steps list
# slider = dict(
#     active=len(y_preds_list_epochs)
#     + 1,  # set the active index to the last trained n.n. plot
#     steps=steps,  # set the steps list
#     currentvalue=dict(prefix="Epoch: "),  # set the prefix for the current value display
#     # customize other properties of the slider as you wish
# )

# # add the slider object to the layout of your figure
# fig.update_layout(sliders=[slider])

# # customize the x-axis ticks and labels
# fig.update_xaxes(
#     tickvals=[0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi],
#     ticktext=["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"],
#     showline=True,
# )
# # make the background of the plot white and remove the grid lines
# fig.update_layout(
#     plot_bgcolor="white",
#     xaxis=dict(showgrid=True, linecolor="black", ticks="outside"),
#     yaxis=dict(showgrid=True, linecolor="black", ticks="outside"),
#     paper_bgcolor="white",
#     font=dict(size=18),
#     title_font_size=24,
#     legend_font_size=20,
#     legend=dict(
#         x=0.6,
#         y=0.9,
#         traceorder="normal",
#         font=dict(family="sans-serif", size=24, color="black"),
#     ),
# )
# fig.update_xaxes()

# # show the figure
# fig.show()
