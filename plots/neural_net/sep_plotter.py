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
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter


# Set y-axis label format to avoid exponential notation
def format_func(value, tick_number):
    return f"{value:.2f}"  # Format to two decimal places


# Define directories for saving plots
fit_plots_dir = os.path.join(os.path.dirname(__file__), "fit_plots")
loss_plots_dir = os.path.join(os.path.dirname(__file__), "loss_plots")

# Create directories if they do not exist
os.makedirs(fit_plots_dir, exist_ok=True)
os.makedirs(loss_plots_dir, exist_ok=True)

# Define filenames
input_filename = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data_q_fwd_stat_6t.csv")
)  # Use absolute path for input file
base_filename = os.path.splitext(os.path.basename(input_filename))[
    0
]  # Get the base name without extension
output_fit_filename = os.path.join(
    fit_plots_dir, f"{base_filename}_fit_muted.png"
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

# Create separate figures for fit and loss plots
fig_fit, ax_fit = plt.subplots(figsize=(8, 6))
fig_loss, ax_loss = plt.subplots(figsize=(8, 6))

# Fit plot
sns.scatterplot(
    x=x_data,
    y=y_data,
    marker="o",
    s=30,
    linewidth=0,
    alpha=0.4,
    color="gray",
    ax=ax_fit,
)

# Add custom legend entry
ax_fit.plot([], [], "o", color="gray", alpha=0.5, markersize=10, label="Training Data")

# Add grid lines for better readability
ax_fit.grid(True, linestyle=":", linewidth=3, alpha=0.3)
x_ticks = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
ax_fit.set_xticks(x_ticks)
ax_fit.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
ax_fit.set_ylabel(r"$f(x)$", fontsize=20, rotation=90, labelpad=29)

# Plot the smoothed lines
sns.lineplot(
    x=x_data_sorted,
    y=y_preds_initial_sorted,
    label="Initial NN",
    linewidth=6,
    alpha=0.8,
    color=palette[0],
    ax=ax_fit,
)

sns.lineplot(
    x=x_data_sorted,
    y=y_preds_trained_sorted,
    label="Trained NN",
    linewidth=6,
    color=palette[1],
    alpha=0.8,
    ax=ax_fit,
)

ax_fit.set_ylim(-3.3, 1.0)
ax_fit.set_xlim(min(x_ticks), max(x_ticks))  # Set fixed x limits for fit plot
ax_fit.legend(fontsize="18", loc="lower right", frameon=True)

# Set spine color for fit plot
for spine in ax_fit.spines.values():
    spine.set_color("black")
    spine.set_linewidth(3)

# Set tick parameters for fit plot
ax_fit.tick_params(axis="both", colors="black", width=3)

# Loss plot
sns.lineplot(
    x=epochs_list,
    y=train_loss_list_epochs,
    label="Training Set",
    markers=True,
    alpha=0.8,
    linewidth=4,
    color=palette[1],
    ax=ax_loss,
)

sns.lineplot(
    x=epochs_list,
    y=test_loss_list_epochs,
    label="Testing Set",
    markers=True,
    linewidth=4,
    alpha=0.8,
    color=palette[0],
    ax=ax_loss,
)

# Set x-axis label and font size for loss plot
ax_loss.set_xlabel("Epoch", fontsize=20)
ax_loss.set_ylabel(
    r"$C_{\mathbf{\theta}}(\mathbf{X}, \mathbf{Y})$",
    fontsize=22,
    rotation=90,
    labelpad=15,
)

# Set y-axis to logarithmic scale for loss plot
ax_loss.set_yscale("log")

# Set fixed y limits for loss plot
ax_loss.set_xlim(min(epochs_list), max(epochs_list))
ax_loss.set_ylim(0.01, 1.2)

# Improve legend appearance for loss plot
ax_loss.legend(loc="upper right", fontsize=18)

# Adjust grid lines for better readability
ax_loss.grid(True, linestyle=":", linewidth=3, alpha=0.3)

# Set spine color for loss plot
for spine in ax_loss.spines.values():
    spine.set_color("black")
    spine.set_linewidth(3)

# Set tick parameters for loss plot
ax_loss.tick_params(axis="both", colors="black", width=3)

# Set y-axis label format to avoid exponential notation
ax_loss.yaxis.set_major_formatter(FuncFormatter(format_func))  # Apply the formatter

# Adjust layout for both plots
fig_fit.tight_layout()
fig_loss.tight_layout()

# Set consistent left margin for both plots
left_margin = 0.2  # Adjust this value as needed
right_margin = 0.2  # Adjust this value as needed
fig_fit.subplots_adjust(left=left_margin)
fig_loss.subplots_adjust(left=left_margin)

# Save the plots separately
fig_fit.savefig(output_fit_filename, dpi=400, bbox_inches="tight")
fig_loss.savefig(output_loss_filename, dpi=400, bbox_inches="tight")

# Close the figures to free up memory
plt.close(fig_fit)
plt.close(fig_loss)

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
