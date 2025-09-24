

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

# style knobs
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE  = 12
YRANGE = (.4, 1)

same_color = "#1f77b4"  # first two
diff_color = "#ff7f0e"  # third

############################################################# Agnostic

datasets = {
    "# no ref, 1.2, pn": {
        "A": [0.67, 0.64, 0.67, 0.61, 0.69, 0.67, 0.70, 0.59, 0.67, 0.64],
        "B": [0.64, 0.43, 0.63, 0.66, 0.60, 0.63, 0.66, 0.62, 0.66, 0.51],
        "Random": [0.58, 0.58, 0.59, 0.48, 0.57, 0.61, 0.63, 0.64, 0.56, 0.49],
    },
    "# no ref, 1.2, 2a": {
        "A": [0.59, 0.72, 0.67, 0.63, 0.72, 0.67, 0.71, 0.68, 0.67, 0.69],
        "B": [0.74, 0.78, 0.83, 0.82, 0.78, 0.83, 0.78, 0.76, 0.78, 0.75],
        "Random": [0.62, 0.62, 0.61, 0.68, 0.63, 0.40, 0.44, 0.44, 0.56, 0.46],
    },
    "# ref, 1.2, pn": {
        "A": [0.67, 0.71, 0.69, 0.58, 0.58, 0.67, 0.71, 0.63, 0.62, 0.57],
        "B": [0.60, 0.54, 0.61, 0.72, 0.51, 0.46, 0.51, 0.61, 0.68, 0.52],
        "C": [0.57, 0.54, 0.56, 0.45, 0.51, 0.54, 0.46, 0.53, 0.51, 0.52],
    },
    "# ref, 1.2, 2a": {
        "A": [0.69, 0.78, 0.69, 0.46, 0.67, 0.72, 0.64, 0.85, 0.68, 0.63],
        "B": [0.75, 0.77, 0.75, 0.83, 0.71, 0.78, 0.75, 0.83, 0.79, 0.70],
        "C": [0.40, 0.57, 0.68, 0.62, 0.70, 0.72, 0.53, 0.41, 0.49, 0.46],
    },
    "# no ref, 1.4, 2a": {
        "A": [0.83, 0.66, 0.86, 0.72, 0.69, 0.66, 0.41, 0.55, 0.72, 0.79],
        "B": [0.76, 0.88, 0.83, 0.84, 0.81, 0.78, 0.79, 0.83, 0.90, 0.83],
        "C": [0.44, 0.59, 0.48, 0.48, 0.54, 0.55, 0.46, 0.55, 0.49, 0.46],
    },
    "# ref, 1.4, 2a": {
        "A": [0.52, 0.72, 0.83, 0.79, 0.59, 0.69, 0.62, 0.66, 0.48, 0.76],
        "B": [0.79, 0.66, 0.81, 0.81, 0.79, 0.79, 0.76, 0.72, 0.88, 0.84],
        "C": [0.46, 0.48, 0.55, 0.44, 0.47, 0.48, 0.45, 0.51, 0.53, 0.46],
    },
    "# no-ref, 1.4, pn": {
        "A": [0.65, 0.65, 0.69, 0.59, 0.60, 0.58, 0.65, 0.68, 0.68, 0.68],
        "B": [0.68, 0.54, 0.62, 0.65, 0.57, 0.57, 0.78, 0.65, 0.56, 0.67],
        "C": [0.58, 0.58, 0.59, 0.48, 0.57, 0.61, 0.63, 0.64, 0.56, 0.49],
    },
    "# ref, 1.4, pn": {
        "A": [0.65, 0.67, 0.73, 0.63, 0.68, 0.61, 0.67, 0.57, 0.68, 0.69],
        "B": [0.44, 0.49, 0.59, 0.52, 0.56, 0.54, 0.62, 0.62, 0.54, 0.49],
        "C": [0.57, 0.54, 0.46, 0.51, 0.41, 0.54, 0.46, 0.53, 0.51, 0.52],
    },
}

# style knobs (doubled font sizes)
TITLE_FONTSIZE = 1636
LABEL_FONTSIZE = 20
TICK_FONTSIZE  = 24
MEAN_FONTSIZE  = 20
YRANGE = (.4, 1)

same_color = "#1f77b4"  # first two
diff_color = "#ff7f0e"  # third

# -------------- plotting loop --------------
for title, data in datasets.items():
    df = pd.DataFrame(data)

    # Print mean and median for each whisker (column)
    print(f"--- {title} ---")
    means = []
    for col in df.columns:
        mean_val = df[col].mean()
        median_val = df[col].median()
        means.append(mean_val)
        print(f"{col}: mean = {mean_val:.3f}, median = {median_val:.3f}")

    # Force x-axis labels
    labels = ["Pattern A", "Pattern B", "Mix"]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    series_list = [df.iloc[:, i].dropna().values for i in range(df.shape[1])]
    bp = ax.boxplot(
        series_list,
        labels=labels,
        patch_artist=True,
        showmeans=False,
        widths=0.3   # smaller widths -> whiskers closer
    )

    # colors: first two same, third different
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(same_color if i < 2 else diff_color)
        box.set_edgecolor("black")
    for whisk in bp["whiskers"]:
        whisk.set_color("black")
    for cap in bp["caps"]:
        cap.set_color("black")
    for med in bp["medians"]:
        med.set_color("black")

    # Add mean values slightly northeast of each box
    for i, mean_val in enumerate(means, start=1):
        ax.text(
            i + 0.2,
            mean_val + 0.03,
            f"μ={mean_val:.2f}",
            ha="left", va="bottom",
            fontsize=MEAN_FONTSIZE, color="black"
        )

    ax.set_ylim(*YRANGE)
    ax.set_ylabel("Accuracy [-]", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"agnostic_{title}", fontsize=TITLE_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.tight_layout()
    fig.savefig(f"agnostic__{title}.svg", format="svg")
    plt.close(fig)

############################################################# Zero training
datasets = {
    "# no ref, 1.2, pn": {
        "A": [0.70, 0.64, 0.65, 0.58, 0.50, 0.53, 0.61, 0.72, 0.64, 0.66],
        "B": [0.62, 0.56, 0.46, 0.57, 0.58, 0.50, 0.50, 0.60, 0.56, 0.47],
        "Random": [0.51, 0.51, 0.43, 0.47, 0.57, 0.51, 0.61, 0.44, 0.53, 0.59],
    },
    "# no ref, 1.2, 2a": {
        "A": [0.61, 0.56, 0.51, 0.43, 0.65, 0.51, 0.74, 0.65, 0.62, 0.75],
        "B": [0.78, 0.85, 0.78, 0.83, 0.66, 0.67, 0.81, 0.65, 0.68, 0.57],
        "Random": [0.44, 0.73, 0.45, 0.61, 0.52, 0.62, 0.52, 0.53, 0.55, 0.47],
    },
    "# ref, 1.2, pn": {
        "A": [0.71, 0.58, 0.58, 0.57, 0.52, 0.56, 0.60, 0.70, 0.56, 0.64],
        "B": [0.76, 0.52, 0.49, 0.59, 0.58, 0.50, 0.58, 0.49, 0.53, 0.46],
        "C": [0.51, 0.53, 0.41, 0.48, 0.51, 0.59, 0.59, 0.50, 0.49, 0.57],
    },
    "# ref, 1.2, 2a": {
        "A": [0.53, 0.58, 0.60, 0.56, 0.62, 0.47, 0.60, 0.61, 0.60, 0.60],
        "B": [0.53, 0.80, 0.77, 0.65, 0.61, 0.65, 0.77, 0.69, 0.66, 0.49],
        "Random": [0.45, 0.71, 0.53, 0.65, 0.57, 0.58, 0.53, 0.57, 0.53, 0.50],
    },

    "# no ref, 1.4, 2a": {
        "A": [0.74, 0.74, 0.69, 0.83, 0.78, 0.71, 0.86, 0.71, 0.88, 0.61],
        "B": [0.68, 0.91, 0.77, 0.88, 0.45, 0.73, 0.00, 0.75, 0.80, 0.73],
        "Random": [0.44, 0.73, 0.45, 0.61, 0.52, 0.63, 0.52, 0.53, 0.55, 0.47],
    },
    "# ref, 1.4, 2a": {
        "A": [float('nan')] * 10,
        "B": [0.86, 0.76, 0.71, 0.69, 0.78, 0.61, 0.85, 0.74, 0.39, 0.71],
        "Random": [0.48, 0.49, 0.51, 0.60, 0.51, 0.58, 0.49, 0.53, 0.51, 0.47],
    },
    "# no-ref, 1.4, pn": {
        "A": [0.59, 0.61, 0.58, 0.65, 0.51, 0.70, 0.59, 0.58, 0.71, 0.70],
        "B": [0.54, 0.59, 0.51, 0.58, 0.57, 0.70, 0.54, 0.52, 0.60, 0.51],
        "Random": [0.51, 0.51, 0.43, 0.47, 0.57, 0.51, 0.61, 0.44, 0.53, 0.59],
    },
    "# ref, 1.4, pn": {
        "A": [0.54, 0.58, 0.57, 0.61, 0.47, 0.70, 0.59, 0.55, 0.68, 0.71],
        "B": [0.53, 0.52, 0.57, 0.49, 0.51, 0.52, 0.48, 0.59, 0.53, 0.52],
        "Random": [0.51, 0.53, 0.41, 0.48, 0.51, 0.59, 0.59, 0.50, 0.49, 0.57],
    },
}


# style knobs (doubled font sizes)
TITLE_FONTSIZE = 1636
LABEL_FONTSIZE = 20
TICK_FONTSIZE  = 24
MEAN_FONTSIZE  = 20
YRANGE = (.4, 1)

same_color = "#1f77b4"  # first two
diff_color = "#ff7f0e"  # third

# -------------- plotting loop --------------
for title, data in datasets.items():
    df = pd.DataFrame(data)

    # Print mean and median for each whisker (column)
    print(f"--- {title} ---")
    means = []
    for col in df.columns:
        mean_val = df[col].mean()
        median_val = df[col].median()
        means.append(mean_val)
        print(f"{col}: mean = {mean_val:.3f}, median = {median_val:.3f}")

    # Force x-axis labels
    labels = ["Pattern A", "Pattern B", "Mix"]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    series_list = [df.iloc[:, i].dropna().values for i in range(df.shape[1])]
    bp = ax.boxplot(
        series_list,
        labels=labels,
        patch_artist=True,
        showmeans=False,
        widths=0.3   # smaller widths -> whiskers closer
    )

    # colors: first two same, third different
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(same_color if i < 2 else diff_color)
        box.set_edgecolor("black")
    for whisk in bp["whiskers"]:
        whisk.set_color("black")
    for cap in bp["caps"]:
        cap.set_color("black")
    for med in bp["medians"]:
        med.set_color("black")

    # Add mean values slightly northeast of each box
    for i, mean_val in enumerate(means, start=1):
        ax.text(
            i + 0.2,
            mean_val + 0.03,
            f"μ={mean_val:.2f}",
            ha="left", va="bottom",
            fontsize=MEAN_FONTSIZE, color="black"
        )

    ax.set_ylim(*YRANGE)
    ax.set_ylabel("Accuracy [-]", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"zero_{title}", fontsize=TITLE_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.tight_layout()
    fig.savefig(f"zero__{title}.svg", format="svg")
    plt.close(fig)
