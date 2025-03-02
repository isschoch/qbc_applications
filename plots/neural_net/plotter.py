import sys
import os

# Add the parent directory to the system path to enable absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Define directories for saving plots
fit_plots_dir = os.path.join(os.path.dirname(__file__), "fit_plots")
loss_plots_dir = os.path.join(os.path.dirname(__file__), "loss_plots")

# Create directories if they do not exist
os.makedirs(fit_plots_dir, exist_ok=True)
os.makedirs(loss_plots_dir, exist_ok=True)

# Define filenames
input_filename = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data_q_fwd_stat_12t.csv")
)  # Use absolute path for input file
base_filename = os.path.splitext(os.path.basename(input_filename))[
    0
]  # Get the base name without extension
output_fit_filename = os.path.join(
    fit_plots_dir, f"{base_filename}_fit.png"
)  # New output filename for muted fit plot
output_loss_filename = os.path.join(
    loss_plots_dir, f"{base_filename}_loss.png"
)  # Configure output filename for loss

df = pd.read_csv(input_filename)
# print(df)
data = df.to_numpy()
epochs_list = np.array(data[2:, 0], dtype=int)
num_epochs = epochs_list[-1] + 1

y_preds_list_epochs = data[2:, 3:]
test_loss_list_epochs = data[2:, 2]
train_loss_list_epochs = data[2:, 1]
x_data = data[0, 3:].flatten()
y_data = data[1, 3:].flatten()

for idx in [0, 50, 100, 150, 200]:
    print("test_loss_list_epochs[", idx, "] =", test_loss_list_epochs[idx])

# Sort the data based on x_data
sorted_indices = np.argsort(x_data)
x_data_sorted = x_data[sorted_indices]
y_data_sorted = y_data[sorted_indices]
y_preds_initial_sorted = y_preds_list_epochs[0][sorted_indices]
y_preds_trained_sorted = y_preds_list_epochs[-1][sorted_indices]

# set global font size
plt.rcParams["font.size"] = 22

# Use a muted color palette
palette = sns.color_palette("muted", 2)  # Use muted colors for better aesthetics

# set figure size
plt.figure(figsize=(8, 6))
ax = sns.scatterplot(
    x=x_data,
    y=y_data,
    marker="o",
    s=30,  # Change markersize to 's' for size
    linewidth=0,
    alpha=0.4,
    color="gray",
    # label="Training Data",
)

# Add custom legend entry
plt.plot([], [], "o", color="gray", alpha=0.5, markersize=10, label="Training Data")

# Add grid lines for better readability
ax.grid(True, linestyle=":", linewidth=3, alpha=0.3)
x_ticks = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
ax.set_xticks(x_ticks)
ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"])
ax.set_xlabel("x", fontsize=20)  # Increased font size for x-axis
ax.set_ylabel(
    "f(x)",
    fontsize=20,
    rotation=0,
    labelpad=25,  # Increased font size for y-axis and rotated by 90 degrees, added padding
)

# Plot the smoothed lines
sns.lineplot(
    x=x_data_sorted,
    y=y_preds_initial_sorted,
    label="Initial NN",
    # linestyle="dotted",
    linewidth=6,
    alpha=0.8,
    color=palette[0],  # Use muted color palette
)

sns.lineplot(
    x=x_data_sorted,
    y=y_preds_trained_sorted,
    label="Trained NN",
    # linestyle="dashed",
    linewidth=6,
    color=palette[1],  # Use muted color palette
    alpha=0.8,
)

plt.ylim(-3.3, 1.0)
plt.xlim(min(x_ticks), max(x_ticks))
# Improve legend appearance
plt.legend(fontsize="18", loc="lower right", frameon=True)

# Adjust layout for better spacing
plt.tight_layout(pad=0.5)  # Increase padding for better spacing
plt.savefig(output_fit_filename, dpi=400)

# set figure size for loss plot
plt.figure(figsize=(8, 6))

# plot train loss and test loss with different line styles and markers
ax = sns.lineplot(
    x=epochs_list,
    y=train_loss_list_epochs,
    label="train cost",
    markers=True,
    linewidth=3,
    color=palette[1],
    data=data,
    alpha=0.8,
)

sns.lineplot(
    x=epochs_list,
    y=test_loss_list_epochs,
    label="test cost",
    markers=True,
    linewidth=3,
    color=palette[0],
    data=data,
)

# set x-axis label and font size
ax.set_xlabel("Epoch", fontsize=16)  # Increased font size for x-axis

# set y-axis label and font size
ax.set_ylabel(
    "Cost $C_{\\mathbf{\\theta}}$", fontsize=16
)  # Increased font size for y-axis

# set legend title, font size, and position
plt.legend(title="Cost Type", loc="upper right")

# adjust spacing
plt.tight_layout(pad=3.0)  # Increase padding for better spacing
plt.ylim(-0.018, 1.218)
plt.savefig(output_loss_filename, dpi=400)

epoch_period = num_epochs // (len(epochs_list) - 1)

# create a figure with plotly express
fig = px.scatter(
    x=x_data,
    y=y_data,
    opacity=0.4,
    color_discrete_sequence=["black"],
    labels={"x": "x", "y": "y"},
)

# add the initial and trained n.n. lines with plotly graph objects
fig.add_trace(
    go.Scatter(
        x=x_data,
        y=y_preds_list_epochs[0],
        mode="lines",
        line=dict(color="blue", dash="dot", width=4),
        name="initial n.n.",
        # set visibility to True by default
        visible=True,
    ),
)

# add traces for each epoch in y_preds_list_epochs
for epoch in epochs_list:
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_preds_list_epochs[epoch // epoch_period],
            mode="lines",
            line=dict(color="red", width=4),
            name=f"Epoch {epoch:4d}, cost={(train_loss_list_epochs[epoch // epoch_period]):.3f}",
            # name=f"Epoch {epoch:4d}",
            # set visibility to False by default
            visible=False,
        )
    )

# create a list of steps for the slider
steps = []
for epoch in epochs_list:
    step = dict(
        method="update",
        args=[
            {"visible": [True, True] + [False] * (len(fig.data) - 2)}
        ],  # set all traces except the first two invisible
        label=f"{epoch}",  # set label for slider step
    )
    # set the (i + 2)th trace (corresponding to the ith epoch) visible
    step["args"][0]["visible"][epoch // epoch_period + 2] = True
    steps.append(step)

# create a slider object with the steps list
slider = dict(
    active=len(y_preds_list_epochs)
    + 1,  # set the active index to the last trained n.n. plot
    steps=steps,  # set the steps list
    currentvalue=dict(prefix="Epoch: "),  # set the prefix for the current value display
    # customize other properties of the slider as you wish
)

# add the slider object to the layout of your figure
fig.update_layout(sliders=[slider])

# customize the x-axis ticks and labels
fig.update_xaxes(
    tickvals=[0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
    ticktext=["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"],
    showline=True,
)
# make the background of the plot white and remove the grid lines
fig.update_layout(
    plot_bgcolor="white",
    xaxis=dict(showgrid=True, linecolor="black", ticks="outside"),
    yaxis=dict(showgrid=True, linecolor="black", ticks="outside"),
    paper_bgcolor="white",
    font=dict(size=18),
    title_font_size=24,
    legend_font_size=20,
    legend=dict(
        x=0.6,
        y=0.9,
        traceorder="normal",
        font=dict(family="sans-serif", size=24, color="black"),
    ),
)
fig.update_xaxes()

# show the figure
# fig.show()
