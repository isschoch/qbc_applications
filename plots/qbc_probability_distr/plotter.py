import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
data = df.to_numpy()
t_wires_range = np.array(data[:, 0], dtype=int)

# set global font size
plt.rcParams["font.size"] = 22

# set figure size and layout
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

lw = 3.5
palette = sns.color_palette("hls", len(t_wires_range))
for idx, num_t_wires in enumerate(t_wires_range):
    x_vals = [
        np.sin(j * np.pi / 2**num_t_wires) ** 2 for j in range(2**num_t_wires)
    ]
    y_vals = data[idx, 1:]
    y_vals = y_vals[np.isfinite(y_vals)]
    # sns.scatterplot(
    #     x=x_vals, y=y_vals, marker=".", linewidth=0, color=palette[idx], ax=axs[0]
    # )
    sns.lineplot(
        x=x_vals,
        y=y_vals,
        linewidth=lw,
        marker=None,
        color=palette[idx],
        ax=axs[1],
        label=f"$t={num_t_wires}$",
    )
# axs[0].legend([f"$t={t}$" for t in t_wires_range])
# state_plot_idx = 0
# y_vals = data[state_plot_idx, 1:]
axs[1].set_xlabel("Approx. Scaled Bin. Inner Prod. $\\hat{f}_b$")

for idx, num_t_wires in enumerate(t_wires_range):
    x_vals = [j * np.pi / 2**num_t_wires for j in range(2**num_t_wires)]
    y_vals = data[idx, 1:]
    y_vals = y_vals[np.isfinite(y_vals)]
    sns.lineplot(
        x=x_vals,
        y=y_vals,
        color=palette[idx],
        linewidth=lw,
        ax=axs[0],
        # label=f"$t={num_t_wires}$",
    )
axs[0].set_xlabel("Approx. Grover Angle $\\hat{\\theta}_{f_b}$")

# Set common y-axis label
fig.text(0.02, 0.6, "Mmt. Prob.", ha="center", va="center", rotation="vertical")

plt.tight_layout()
plt.subplots_adjust(left=0.08)
# plt.show()
plt.savefig("qbc_probability_distr.png", dpi=400)


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("data.csv")
# data = df.to_numpy()
# t_wires_range = np.array(data[:, 0], dtype=int)

# # set global font size
# plt.rcParams["font.size"] = 22

# # set figure size
# plt.figure(figsize=(8, 6))
# lw = 2
# palette = sns.color_palette("hls", len(t_wires_range))
# for idx, num_t_wires in enumerate(t_wires_range):
#     x_vals = [
#         np.sin(j * np.pi / 2**num_t_wires) ** 2 for j in range(2**num_t_wires)
#     ]
#     y_vals = data[idx, 1:]
#     y_vals = y_vals[np.isfinite(y_vals)]
#     sns.scatterplot(x=x_vals, y=y_vals, marker=".", linewidth=0, color=palette[idx])
#     sns.lineplot(x=x_vals, y=y_vals, linewidth=lw, marker=None, color=palette[idx])
# plt.legend([f"$t={t}$" for t in t_wires_range])
# plt.ylabel("Mmt. Prob.")
# plt.xlabel("Approx. Grover Angle $\\hat{\\theta}_{f_b}$")

# plt.tight_layout()
# plt.show()

# plt.xlabel("State $j$")
# plt.tight_layout()
# sns.lineplot(
#     x=range(2 ** t_wires_range[-1]), y=data[-1, 1:], color=palette[-1], linewidth=lw
# )
# plt.show()

# y_preds_list_epochs = data[2:, 3:]
# test_loss_list_epochs = data[2:, 2]
# train_loss_list_epochs = data[2:, 1]
# x_data = data[0, 3:].flatten()
# y_data = data[1, 3:].flatten()


# # set global font size
# plt.rcParams["font.size"] = 22

# # set figure size
# plt.figure(figsize=(8, 6))

# ax = sns.scatterplot(
#     x=x_data,
#     y=y_data,
#     marker=".",
#     linewidth=0,
#     alpha=0.4,
#     color="black",
#     label="data",
# )
# ax.set_xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
# ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"])
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# sns.lineplot(
#     x=x_data,
#     y=y_preds_list_epochs[0],
#     label="initial n.n.",
#     linestyle="dotted",
#     linewidth=4,
# )


# # for y_preds, epoch_idx in zip(y_preds_list_epochs, epochs_list[2:]):
# #     sns.lineplot(
# #         x=x_data,
# #         y=y_preds,
# #         # label="trained n.n.",
# #         linestyle="dashed",
# #         linewidth=4,
# #         # color="red",
# #         hue=epoch_idx / num_epochs,
# #         alpha=epoch_idx**2 / num_epochs**2,
# #         palette="Set1",
# #     )

# sns.lineplot(
#     x=x_data,
#     y=y_preds_list_epochs[-1],
#     label="trained n.n.",
#     linestyle="dashed",
#     linewidth=4,
#     color="red",
# )
# plt.tight_layout()
# plt.show()
# # plt.savefig("neural_net_fit_q_rev.png", dpi=400)

# # set figure size
# plt.figure(figsize=(8, 6))

# # # plot train loss
# # ax = sns.lineplot(
# #     x=epochs_list,
# #     y=train_loss_list_epochs,
# #     label="train loss",
# #     linestyle="dotted",
# #     linewidth=4,
# #     color="red",
# # )

# # # plot test loss
# # sns.lineplot(
# #     x=epochs_list,
# #     y=test_loss_list_epochs,
# #     label="test loss",
# #     # linestyle="dotted",
# #     linewidth=4,
# #     color="green",
# #     alpha=0.8,
# # )

# palette = sns.color_palette("hls", 2)

# # plot train loss and test loss with different line styles and markers
# ax = sns.lineplot(
#     x=epochs_list,
#     y=train_loss_list_epochs,
#     label="train cost",
#     markers=True,
#     linewidth=3,
#     color=palette[1],
#     data=data,
#     alpha=0.8,
# )

# sns.lineplot(
#     x=epochs_list,
#     y=test_loss_list_epochs,
#     label="test cost",
#     markers=True,
#     linewidth=3,
#     color=palette[0],
#     data=data,
# )

# # set x-axis label and font size
# ax.set_xlabel("Epoch")

# # set y-axis label and font size
# ax.set_ylabel("Cost $C_{\\mathbf{\\theta}}$")


# # set legend title, font size, and position
# plt.legend(title="Cost Type", loc="upper right")

# # adjust spacing
# plt.tight_layout()

# # show plot
# plt.show()
# # plt.savefig("neural_net_loss_q_rev.png", dpi=400)

# import plotly.express as px
# import plotly.graph_objects as go

# epoch_period = num_epochs // (len(epochs_list) - 1)

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
#             name=f"Epoch {epoch:4d}, cost={(train_loss_list_epochs[epoch // epoch_period]):.3f}",
#             # name=f"Epoch {epoch:4d}",
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
#     tickvals=[0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
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
