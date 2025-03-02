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
from matplotlib.gridspec import GridSpec


# Set y-axis label format to avoid exponential notation
def format_func(value, tick_number):
    return f"{value:.1f}"  # Format to two decimal places


# Define directories for saving plots
comb_plots_dir = os.path.join(os.path.dirname(__file__), "comb_plots")

# Create directories if they do not exist
os.makedirs(comb_plots_dir, exist_ok=True)

# Define filenames for two input CSV files
input_filename_1 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data_q_fwd_stat_6t.csv")
)  # First input file
input_filename_2 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data_q_fwd_stat_12t.csv")
)  # Second input file

# Define output filename
base_filename = "combined_plot"
output_comb_filename = os.path.join(
    comb_plots_dir, f"{base_filename}_comb.png"
)  # New output filename for muted fit plot

# Read data from both CSV files
df1 = pd.read_csv(input_filename_1)
df2 = pd.read_csv(input_filename_2)

# Convert data to numpy arrays
data1 = df1.to_numpy()
data2 = df2.to_numpy()

# Extract relevant data for the first case
epochs_list_1 = np.array(data1[2:, 0], dtype=int)
train_loss_list_epochs_1 = data1[2:, 1]
test_loss_list_epochs_1 = data1[2:, 2]
x_data_1 = data1[0, 3:].flatten()
y_data_1 = data1[1, 3:].flatten()
y_preds_list_epochs_1 = data1[2:, 3:]

# Extract relevant data for the second case
epochs_list_2 = np.array(data2[2:, 0], dtype=int)
train_loss_list_epochs_2 = data2[2:, 1]
test_loss_list_epochs_2 = data2[2:, 2]
x_data_2 = data2[0, 3:].flatten()
y_data_2 = data2[1, 3:].flatten()
y_preds_list_epochs_2 = data2[2:, 3:]

# Sort the data based on x_data for both cases
sorted_indices_1 = np.argsort(x_data_1)
sorted_indices_2 = np.argsort(x_data_2)
x_data_sorted_1 = x_data_1[sorted_indices_1]
y_data_sorted_1 = y_data_1[sorted_indices_1]
y_preds_initial_sorted_1 = y_preds_list_epochs_1[0][sorted_indices_1]
y_preds_trained_sorted_1 = y_preds_list_epochs_1[-1][sorted_indices_1]

x_data_sorted_2 = x_data_2[sorted_indices_2]
y_data_sorted_2 = y_data_2[sorted_indices_2]
y_preds_initial_sorted_2 = y_preds_list_epochs_2[0][sorted_indices_2]
y_preds_trained_sorted_2 = y_preds_list_epochs_2[-1][sorted_indices_2]

# set global font size
plt.rcParams["font.size"] = 22

# Use a muted color palette
palette = [
    sns.color_palette("muted")[1],
    sns.color_palette("muted")[0],
]  # [orange, blue]

# Create a figure with a gridspec layout
fig = plt.figure(figsize=(16, 14))  # Increased figure height
gs = GridSpec(
    2, 2, height_ratios=[1, 1.0]
)  # 2 rows, 2 columns, with custom height ratio

# Create subplots using the gridspec
ax_fit1 = fig.add_subplot(gs[0, 0])
ax_fit2 = fig.add_subplot(gs[0, 1])
ax_loss = fig.add_subplot(gs[1, :])  # Span both columns in the second row

# Fit plot for the first case
sns.scatterplot(
    x=x_data_1,
    y=y_data_1,
    marker="o",
    s=30,
    linewidth=0,
    alpha=0.4,
    color="gray",
    ax=ax_fit1,
)

# Add custom legend entry for the first case
ax_fit1.plot([], [], "o", color="gray", alpha=0.5, markersize=10, label="Training Data")

# Add grid lines for better readability
ax_fit1.grid(True, linestyle=":", linewidth=3, alpha=0.3)
x_ticks = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
ax_fit1.set_xticks(x_ticks)
ax_fit1.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
ax_fit1.set_ylabel(r"$f_{\mathbf{\theta}}(x)$", fontsize=20, rotation=90, labelpad=15)

ax_fit1.set_yticks(range(-3, 2))
ax_fit1.set_title(r"$t=6$ Ancillae Qubits", fontsize=20, pad=20)

# Plot the smoothed lines for the first case (orange)
sns.lineplot(
    x=x_data_sorted_1,
    y=y_preds_initial_sorted_1,
    label=r"Initial $f_{\mathbf{\theta}}(x)$",
    linewidth=6,
    alpha=0.8,
    color=sns.light_palette(palette[0], as_cmap=False)[2],  # Lighter orange
    linestyle="--",  # Dashed line for initial NN
    ax=ax_fit1,
)

sns.lineplot(
    x=x_data_sorted_1,
    y=y_preds_trained_sorted_1,
    label=r"Trained $f_{\mathbf{\theta}}(x)$",
    linewidth=6,
    color=palette[0],  # Solid orange
    alpha=0.8,
    ax=ax_fit1,
)

ax_fit1.set_ylim(-3.0, 1.0)
ax_fit1.set_xlim(min(x_ticks), max(x_ticks))  # Set fixed x limits for fit plot
ax_fit1.legend(fontsize="18", loc="lower right", frameon=True)

# Set spine color for fit plot
for spine in ax_fit1.spines.values():
    spine.set_color("black")
    spine.set_linewidth(3)

# Set tick parameters for fit plot
ax_fit1.tick_params(axis="both", colors="black", width=3)

# Fit plot for the second case
sns.scatterplot(
    x=x_data_2,
    y=y_data_2,
    marker="o",
    s=30,
    linewidth=0,
    alpha=0.4,
    color="gray",
    ax=ax_fit2,
)

# Add custom legend entry for the second case
ax_fit2.plot([], [], "o", color="gray", alpha=0.5, markersize=10, label="Training Data")

# Add grid lines for better readability
ax_fit2.grid(True, linestyle=":", linewidth=3, alpha=0.3)
ax_fit2.set_xticks(x_ticks)
ax_fit2.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])

ax_fit2.set_yticks(range(-3, 2))
ax_fit2.set_title(r"$t=12$ Ancillae Qubits", fontsize=20, pad=20)

# Plot the smoothed lines for the second case (blue)
sns.lineplot(
    x=x_data_sorted_2,
    y=y_preds_initial_sorted_2,
    label=r"Initial $f_{\mathbf{\theta}}(x)$",
    linewidth=6,
    alpha=0.8,
    color=sns.light_palette(palette[1], as_cmap=False)[2],  # Lighter blue
    linestyle="--",  # Dashed line for initial NN
    ax=ax_fit2,
)

sns.lineplot(
    x=x_data_sorted_2,
    y=y_preds_trained_sorted_2,
    label=r"Trained $f_{\mathbf{\theta}}(x)$",
    linewidth=6,
    color=palette[1],  # Solid blue
    alpha=0.8,
    ax=ax_fit2,
)
ax_fit2.set_ylim(-3.0, 1.0)
ax_fit2.set_xlim(min(x_ticks), max(x_ticks))  # Set fixed x limits for fit plot
ax_fit2.legend(fontsize="18", loc="lower right", frameon=True)
ax_fit2.set_yticklabels([])  # Remove y-axis labels while keeping the ticks

# Set spine color for fit plot
for spine in ax_fit2.spines.values():
    spine.set_color("black")
    spine.set_linewidth(3)

# Set tick parameters for fit plot
ax_fit2.tick_params(axis="both", colors="black", width=3)

# # Loss plot
# sns.lineplot(
#     x=epochs_list_1,
#     y=train_loss_list_epochs_1,
#     label="Training (6 Ancillae)",
#     markers=True,
#     linewidth=4,
#     color=sns.light_palette(palette[0], as_cmap=False)[2],
#     linestyle="--",  # Dashed line for training set
#     ax=ax_loss,
# )

sns.lineplot(
    x=epochs_list_1,
    y=test_loss_list_epochs_1,
    label=r"$t=6$ Ancillae Qubits",
    markers=True,
    linewidth=4,
    color=palette[0],
    ax=ax_loss,
)

# sns.lineplot(
#     x=epochs_list_2,
#     y=train_loss_list_epochs_2,
#     label="Training (12 Ancillae)",
#     markers=True,
#     alpha=0.8,
#     linewidth=4,
#     color=sns.light_palette(palette[1], as_cmap=False)[2],
#     linestyle="--",  # Dashed line for training set
#     ax=ax_loss,
# )

sns.lineplot(
    x=epochs_list_2,
    y=test_loss_list_epochs_2,
    label=r"$t=12$ Ancillae Qubits",
    markers=True,
    alpha=0.8,
    linewidth=4,
    color=palette[1],
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
ax_loss.set_title(r"Loss Function", fontsize=20, pad=20)
# Set y-axis to logarithmic scale for loss plot
# ax_loss.set_yscale("log")

# Set fixed y limits for loss plot
ax_loss.set_xlim(min(epochs_list_1), max(epochs_list_1))
ax_loss.set_ylim(0.00, 1.0)
ax_loss.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

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

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1)  # Adjust padding between subplots

# Save the combined plot
plt.savefig(output_comb_filename, dpi=400, bbox_inches="tight")

# Close the figure to free up memory
plt.close(fig)
