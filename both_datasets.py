import os
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import glob
from sklearn.utils import resample
import matplotlib.ticker as mticker

# ── Config ─────────────────────────────────────────────
path1 = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\BCI_restructured\Clustering\clustering\2a (exploratory)"
path2 = r"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\BCI_restructured\Clustering\clustering\PhysioNet (exploratory)"
OUT_DIR = Path(__file__).parent.resolve()
os.makedirs(OUT_DIR, exist_ok=True)

# ###################################################################################################### Statistical Results
# # ── Stats ──────────────────────────────────────────────
# f1 = f"{path1}\\ers_values_dataset2.csv"
# f2 = f"{path2}\\ers_values_dataset1.csv"
# e_f1 = f"{path1}\\e_values_dataset2.csv"
# e_f2 = f"{path2}\\e_values_dataset1.csv"
# df1, df2 = pd.read_csv(f1), pd.read_csv(f2)
# e_df1, e_df2 = pd.read_csv(e_f1), pd.read_csv(e_f2)
# ALIASES = {
#     "ers_rl_A":    ["ers_rl_A", "ers_rh_A", "ers_c4_rl_A"],
#     "ers_rl_B":    ["ers_rl_B", "ers_rh_B", "ers_c4_rl_B"],
#     "ers_rl_weak": ["ers_rl_weak", "ers_rh_weak", "ers_c4_rl_weak"],
#     "ers_ll_A":    ["ers_ll_A", "ers_lh_A", "ers_c3_ll_A"],
#     "ers_ll_B":    ["ers_ll_B", "ers_lh_B", "ers_c3_ll_B"],
#     "ers_ll_weak": ["ers_ll_weak", "ers_lh_weak", "ers_c3_ll_weak"],
# }
# eALIASES = {
#     "e_rl_A":    ["e_rl_A", "e_rh_A", "e_c4_rl_A"],
#     "e_rl_B":    ["e_rl_B", "e_rh_B", "e_c4_rl_B"],
#     "e_rl_weak": ["e_rl_weak", "e_rh_weak", "e_c4_rl_weak"],
#     "e_ll_A":    ["e_ll_A", "e_lh_A", "e_c3_ll_A"],
#     "e_ll_B":    ["e_ll_B", "e_lh_B", "e_c3_ll_B"],
#     "e_ll_weak": ["e_ll_weak", "e_lh_weak", "e_c3_ll_weak"],
# }
# def extract_data(df):
#     data = {}
#     for label, options in ALIASES.items():
#         for key in options:
#             if key in df.columns:
#                 data[label] = df[key]
#                 break
#     return data
# def e_extract_data(df):
#     data = {}
#     for label, options in eALIASES.items():
#         for key in options:
#             if key in df.columns:
#                 data[label] = df[key]
#                 break
#     return data

# data1, data2 = extract_data(df1), extract_data(df2)
# e_data1, e_data2 = e_extract_data(e_df1), e_extract_data(e_df2)


# def run_test(label, x, y, n_boot=1000, ci=95):
#     # Drop NaNs
#     x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
#     n1, n2 = len(x), len(y)

#     # Mann–Whitney U test
#     res = mannwhitneyu(x, y, alternative="greater")

#     # Effect size (rank-biserial correlation)
#     r_rb =  (2 * res.statistic) / (n1 * n2) -1

#     # Bootstrap CI for effect size
#     effects = []
#     for _ in range(n_boot):
#         xb = resample(x, replace=True, n_samples=n1)
#         yb = resample(y, replace=True, n_samples=n2)
#         U_boot = mannwhitneyu(xb, yb, alternative="greater").statistic
#         effects.append(1 - (2 * U_boot) / (n1 * n2))
#     lower = np.percentile(effects, (100 - ci) / 2)
#     upper = np.percentile(effects, 100 - (100 - ci) / 2)

#     # Descriptive stats
#     def describe(arr):
#         return np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)

#     x_med, x_q1, x_q3 = describe(x)
#     y_med, y_q1, y_q3 = describe(y)

#     # Print all results
#     print(f"{label}:")
#     print(f"  U = {res.statistic:.1f}, p = {res.pvalue:.3g}")
#     print(f"  Effect size (rank-biserial r_rb) = {r_rb:.3f}, "
#           f"95% CI [{lower:.3f}, {upper:.3f}]")
#     print(f"  {label} group X: median = {x_med:.3f}, IQR = [{x_q1:.3f}, {x_q3:.3f}]")
#     print(f"  {label} group Y: median = {y_med:.3f}, IQR = [{y_q1:.3f}, {y_q3:.3f}]")
# run_test("Dataset1 RL A > weak", data1["ers_rl_A"], data1["ers_rl_weak"])
# run_test("Dataset1 LL B > weak", data1["ers_ll_B"], data1["ers_ll_weak"])
# run_test(" extra Dataset1 RL A > LL A", data1["ers_rl_A"], data1["ers_ll_A"])
# run_test("extra Dataset1 LL B > RL B", data1["ers_ll_B"], data1["ers_rl_B"])


# # run_test("Dataset2 RL A > weak", data2["ers_rl_A"], data2["ers_rl_weak"])
# # run_test("Dataset2 LL B > weak", data2["ers_ll_B"], data2["ers_ll_weak"])
# # run_test(" extra Dataset2 RL A > LL A", data2["ers_rl_A"], data2["ers_ll_A"])
# # run_test("extra Dataset2 LL B > RL B", data2["ers_ll_B"], data2["ers_rl_B"])


# rl_A_all = pd.concat([data1["ers_rl_A"], data2["ers_rl_A"]])
# rl_B_all = pd.concat([data1["ers_rl_B"], data2["ers_rl_B"]])
# rl_weak_all = pd.concat([data1["ers_rl_weak"], data2["ers_rl_weak"]])
# ll_A_all = pd.concat([data1["ers_ll_A"], data2["ers_ll_A"]])
# ll_B_all = pd.concat([data1["ers_ll_B"], data2["ers_ll_B"]])
# ll_weak_all = pd.concat([data1["ers_ll_weak"], data2["ers_ll_weak"]])

# e_rl_A_all = pd.concat([e_data1["e_rl_A"], e_data2["e_rl_A"]])
# e_rl_B_all = pd.concat([e_data1["e_rl_B"], e_data2["e_rl_B"]])
# e_rl_weak_all = pd.concat([e_data1["e_rl_weak"], e_data2["e_rl_weak"]])
# e_ll_A_all = pd.concat([e_data1["e_ll_A"], e_data2["e_ll_A"]])
# e_ll_B_all = pd.concat([e_data1["e_ll_B"], e_data2["e_ll_B"]])
# e_ll_weak_all = pd.concat([e_data1["e_ll_weak"], e_data2["e_ll_weak"]])

# run_test("Combined RL A > weak", rl_A_all, rl_weak_all)
# run_test("Combined LL B > weak", ll_B_all, ll_weak_all)
# run_test("extra Combined RL A > LL A", rl_A_all, ll_A_all)
# run_test("extra Combined LL B > RL B", ll_B_all, rl_B_all)

# def whisker_plot(col1, col2, labels=("DataFrame 1", "DataFrame 2"), save_dir=".", filename=None):
#     """Whisker plots for two datasets with auto-incremented pattern (A, B, ...), saved as SVG."""

#     # Initialize call counter
#     if not hasattr(whisker_plot, "call_count"):
#         whisker_plot.call_count = 0

#     # Assign pattern automatically
#     patterns = ["A", "B", "C", "D"]
#     pattern = patterns[whisker_plot.call_count % len(patterns)]
#     whisker_plot.call_count += 1

#     # Clean NaNs
#     col1 = np.asarray(col1, dtype=float)
#     col2 = np.asarray(col2, dtype=float)
#     col1 = col1[~np.isnan(col1)]
#     col2 = col2[~np.isnan(col2)]
#     data = [col1, col2]

#     fig, ax = plt.subplots(figsize=(6, 4))

#     # Boxplot
#     bp = ax.boxplot(
#         data, labels=labels, showmeans=True, meanline=False,
#         patch_artist=True,
#         medianprops=dict(color="red", linewidth=2),
#         whiskerprops=dict(color="black"),
#         capprops=dict(color="black"),
#         flierprops=dict(marker="o", color="black", alpha=0.5)
#     )

#     # Colors: first = yellow, second = blue (with transparency)
#     colors = ["yellow", "blue"]
#     for patch, color in zip(bp["boxes"], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.5)  # 0.5 = semi-transparent

#     # Label per pattern
#     ax.set_ylabel(f"ERS values for Pattern {pattern}", fontsize=14)

#     # Styling
#     ax.tick_params(axis="both", labelsize=13)
#     ax.grid(True, linestyle="--", alpha=0.6)
#     fig.tight_layout()

#     # Build filename
#     safe_pattern = str(pattern).replace(" ", "_")
#     if filename is None:
#         filename = f"whisker_pattern_{safe_pattern}.svg"
#     out_path = os.path.join(save_dir, filename)

#     # Save as SVG
#     fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
#     plt.close(fig)

#     return out_path
# def whisker_plot2(col1, col2, labels=("DataFrame 1", "DataFrame 2"), save_dir=".", filename=None):
#     """Whisker plots for two datasets with auto-assigned ylabel (LL-MI on 1st call, RL-MI on 2nd)."""

#     # Initialize call counter
#     if not hasattr(whisker_plot2, "call_count"):
#         whisker_plot2.call_count = 0

#     # Assign ylabel automatically
#     ylabels = ["LL-MI", "RL-MI"]
#     current_label = ylabels[whisker_plot2.call_count % len(ylabels)]
#     whisker_plot2.call_count += 1

#     # Clean NaNs
#     col1 = np.asarray(col1, dtype=float)
#     col2 = np.asarray(col2, dtype=float)
#     col1 = col1[~np.isnan(col1)]
#     col2 = col2[~np.isnan(col2)]
#     data = [col1, col2]

#     fig, ax = plt.subplots(figsize=(6, 4))

#     # Boxplot
#     bp = ax.boxplot(
#         data, labels=labels, showmeans=True, meanline=False,
#         patch_artist=True,
#         medianprops=dict(color="red", linewidth=2),
#         whiskerprops=dict(color="black"),
#         capprops=dict(color="black"),
#         flierprops=dict(marker="o", color="black", alpha=0.5)
#     )

#     # Colors: first = yellow, second = blue (with transparency)
#     colors = ["yellow", "blue"]
#     for patch, color in zip(bp["boxes"], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.5)

#     # Y-label for LL-MI or RL-MI
#     ax.set_ylabel(f"ERS values for {current_label}", fontsize=14)

#     # Styling
#     ax.tick_params(axis="both", labelsize=13)
#     ax.grid(True, linestyle="--", alpha=0.6)
#     fig.tight_layout()

#     # Build filename
#     safe_label = current_label.replace(" ", "_")
#     if filename is None:
#         filename = f"whisker_{safe_label}.svg"
#     out_path = os.path.join(save_dir, filename)

#     # Save as SVG
#     fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
#     plt.close(fig)

#     return out_path

# # Example calls
# whisker_plot(rl_A_all, ll_A_all, labels=("RL-MI", "LL-MI"))  # Pattern A
# whisker_plot(ll_B_all, rl_B_all, labels=("LL-MI", "RL-MI"))  # Pattern B
# run_test("RL A vs LL A", rl_A_all, ll_A_all) # should show some significance
# run_test("LL B vs RL B", ll_B_all, rl_B_all) # should show some significance


# whisker_plot2(ll_A_all, ll_weak_all, labels=("Pattern A", "Pattern Weak"))
# whisker_plot2(rl_B_all, rl_weak_all, labels=("Pattern B", "Pattern Weak"))
# run_test("LL A vs LL weak", ll_A_all, ll_weak_all) # should be less significant than Fig 9
# run_test("RL B vs RL weak", rl_B_all, rl_weak_all) # should be less significant than Fig 9




# run_test("e_Combined RL A > weak", e_rl_A_all, e_rl_weak_all)
# run_test("e_Combined LL B > weak", e_ll_B_all, e_ll_weak_all)
# run_test("e_extra Combined RL A > LL A", e_rl_A_all, e_ll_A_all)
# run_test("e_extra Combined LL B > RL B", e_ll_B_all, e_rl_B_all)

# ###################################################################################################### ERS whisker plotting
# # Colors
# color_A = "skyblue"
# color_Weak = "orange"

# # RL plot
# bp = plt.boxplot(
#     [rl_A_all.dropna(), rl_weak_all.dropna()],
#     labels=["Pattern A", "Pattern Weak"],
#     showmeans=True,
#     showfliers=False,         # don't show outliers as points
#     whis=[0, 100],            # whiskers span full data range
#     patch_artist=True
# )

# for i, color in enumerate([color_A, color_Weak]):
#     bp["whiskers"][2*i].set_color("black")
#     bp["whiskers"][2*i+1].set_color("black")
#     bp["caps"][2*i].set_color("black")
#     bp["caps"][2*i+1].set_color("black")
#     bp["boxes"][i].set_edgecolor("black")
#     bp["boxes"][i].set_facecolor(color)
#     bp["medians"][i].set_color("black")
#     bp["means"][i].set_markerfacecolor("black")
#     bp["means"][i].set_markeredgecolor(color)

# plt.ylabel("ERS values for RL-MI", fontsize=14)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.savefig(OUT_DIR / "combined_RL.svg")
# plt.close()


# # LL plot
# bp = plt.boxplot(
#     [ll_B_all.dropna(), ll_weak_all.dropna()],
#     labels=["Pattern B", "Pattern Weak"],
#     showmeans=True,
#     showfliers=False,         # don't show outliers separately
#     whis=[0, 100],            # whiskers span full data range
#     patch_artist=True
# )

# for i, color in enumerate([color_A, color_Weak]):
#     bp["whiskers"][2*i].set_color("black")
#     bp["whiskers"][2*i+1].set_color("black")
#     bp["caps"][2*i].set_color("black")
#     bp["caps"][2*i+1].set_color("black")
#     bp["boxes"][i].set_edgecolor("black")
#     bp["boxes"][i].set_facecolor(color)
#     bp["medians"][i].set_color("black")
#     bp["means"][i].set_markerfacecolor("black")
#     bp["means"][i].set_markeredgecolor(color)

# plt.ylabel("ERS values for LL-MI", fontsize=14)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.savefig(OUT_DIR / "combined_LL.svg")
# plt.close()

# ###################################################################################################### Energy whisker plotting
# # Colors
# color_A = "skyblue"
# color_Weak = "orange"

# # RL plot
# bp = plt.boxplot(
#     [e_rl_A_all.dropna(), e_rl_weak_all.dropna()],
#     labels=["Pattern A", "Pattern Weak"],
#     showfliers=False,         # don't show outliers as points
#     showmeans=True,
#     patch_artist=True
# )

# for i, color in enumerate([color_A, color_Weak]):
#     bp["whiskers"][2*i].set_color("black")
#     bp["whiskers"][2*i+1].set_color("black")
#     bp["caps"][2*i].set_color("black")
#     bp["caps"][2*i+1].set_color("black")
#     bp["boxes"][i].set_edgecolor("black")
#     bp["boxes"][i].set_facecolor(color)
#     bp["medians"][i].set_color("black")
#     bp["means"][i].set_markerfacecolor("black")
#     bp["means"][i].set_markeredgecolor(color)

# plt.ylabel("Energy ratio values for RL-MI", fontsize=14)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.savefig(OUT_DIR / "e_combined_RL.svg")
# plt.close()


# # LL plot
# bp = plt.boxplot(
#     [e_ll_B_all.dropna(), e_ll_weak_all.dropna()],
#     labels=["Pattern B", "Pattern Weak"],
#     showfliers=False,         # don't show outliers as points
#     showmeans=True,
#     patch_artist=True
# )

# for i, color in enumerate([color_A, color_Weak]):
#     bp["whiskers"][2*i].set_color("black")
#     bp["whiskers"][2*i+1].set_color("black")
#     bp["caps"][2*i].set_color("black")
#     bp["caps"][2*i+1].set_color("black")
#     bp["boxes"][i].set_edgecolor("black")
#     bp["boxes"][i].set_facecolor(color)
#     bp["medians"][i].set_color("black")
#     bp["means"][i].set_markerfacecolor("black")
#     bp["means"][i].set_markeredgecolor(color)

# plt.ylabel("Energy ratio values for LL-MI", fontsize=14)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.savefig(OUT_DIR / "e_combined_LL.svg")
# plt.close()

# ###################################################################################################### Printing info about ERS
# def summarize_data(data, label):
#     data = pd.Series(data).dropna()
#     mean_val = data.mean()
#     median_val = data.median()
#     pct_pos = (data > 0).mean() * 100
#     pct_neg = (data < 0).mean() * 100
#     return {
#         "Label": label,
#         "Mean": mean_val,
#         "Median": median_val,
#         "% > 0": pct_pos,
#         "% < 0": pct_neg
#     }

# # RL sets
# rl_A_stats = summarize_data(rl_A_all, "RL - pattern A")
# rl_weak_stats = summarize_data(rl_weak_all, "RL - pattern Weak")
# rl_Bweak_stats = summarize_data(pd.concat([rl_weak_all, rl_B_all]), "RL - pattern B and Weak")

# # LL sets
# ll_B_stats = summarize_data(ll_B_all, "LL - pattern B")
# ll_weak_stats = summarize_data(ll_weak_all, "LL - pattern Weak")
# ll_Aweak_stats = summarize_data(pd.concat([ll_weak_all, ll_A_all]), "LL - pattern A and Weak")

# # Combine into a table
# results = pd.DataFrame([rl_A_stats, rl_weak_stats, rl_Bweak_stats,
#                         ll_B_stats, ll_weak_stats, ll_Aweak_stats])

# print(results)

# # Optional: Save as CSV for later use
# results.to_csv(OUT_DIR / "boxplot_stats.csv", index=False)

# ###################################################################################################### Plotting MI datapoints, whiskers for angles and distances

# ── MI Plots ───────────────────────────────────────────
csv_paths = [Path(path1) / "MI_data.csv", Path(path2) / "MI_data.csv"]
by_group = {k: {"pairs": [], "angles": [], "distances": []} for k in ["subs_pattern_A", "subs_pattern_B", "subs_weak"]}
csv1_resolved = Path(path1).resolve()

for path in csv_paths:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row.get("group", "")
            if g not in by_group: continue
            x0, y0 = float(row["x_LH"]), float(row["y_LH"])
            x1, y1 = float(row["x_RH"]), float(row["y_RH"])
            th, d = float(row["angle_deg"]), float(row["distance"])
            if path.parent.resolve() == csv1_resolved:
                x0 *= 1e-6; y0 *= 1e-6; x1 *= 1e-6; y1 *= 1e-6; d *= 1e-6
            by_group[g]["pairs"].append((x0, y0, x1, y1))
            by_group[g]["angles"].append(th)
            by_group[g]["distances"].append(d)



for gkey, group in by_group.items():
    if not group["pairs"]:
        continue
    plt.figure(figsize=(8, 6))
    
    for idx, (x0, y0, x1, y1) in enumerate(group["pairs"]):
        # rescale to 1e-12 units
        x0 /= 1e-12
        y0 /= 1e-12
        x1 /= 1e-12
        y1 /= 1e-12

        # scatter the two points
        plt.scatter(x0, y0, c=["#1f77b4"], s=200, marker='*',
                    edgecolors='black', linewidths=1)
        plt.scatter(x1, y1, c=["#ff7f0e"], s=200, marker='*',
                    edgecolors='black', linewidths=1)
        plt.plot([x0, x1], [y0, y1], color="gray", linewidth=1, alpha=0.7)

        # angle for this pair (in degrees)
        ang_deg = f"{group['angles'][idx]:.1f}°"

        # midpoint of the segment
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2

        # offset (north-right)
        dx = 0.1 * (plt.xlim()[1] - plt.xlim()[0])
        dy = 0.05 * (plt.ylim()[1] - plt.ylim()[0])

        plt.text(xm + dx, ym + dy, ang_deg, fontsize=14,
                 ha="left", va="bottom", color="black")

    # axes labels (with µV²/Hz, values in 10^-12 scale)
    plt.xlabel(r"$\mathrm{PSD}_{\mathrm{MI,\ LH}}\ (\ \mu V^2/Hz)$", fontsize=20)
    plt.ylabel(r"$\mathrm{PSD}_{\mathrm{MI,\ RH}}\ (\ \mu V^2/Hz)$", fontsize=20)

    # Set axis limits and ticks
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.xticks(range(0, 5), fontsize=14)
    plt.yticks(range(0, 5), fontsize=14)

    # Make tick markers bigger
    plt.tick_params(axis='both', which='major', length=8, width=2)
    plt.tick_params(axis='both', which='minor', length=5, width=1)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"MI_combined_{gkey}.svg")
    plt.close()


# # Whisker plots
# data, labels = [], []
# for k, lab in [("subs_pattern_A", "A"), ("subs_pattern_B", "B"), ("subs_weak", "Weak")]:
#     vals = by_group[k]["angles"]
#     if vals:
#         data.append(np.asarray(vals, float))
#         labels.append(lab)

# # collect data + labels only for groups that have angles
# data, labels = [], []
# for k, lab in [("subs_pattern_A", "A"), ("subs_pattern_B", "B"), ("subs_weak", "Weak")]:
#     vals = by_group[k]["angles"]
#     if vals:
#         data.append(np.asarray(vals, float))
#         labels.append(lab)

# if data:
#     fig, ax = plt.subplots(figsize=(6, 5))

#     bp = ax.boxplot(
#         data,
#         labels=labels,
#         showmeans=True,
#         meanprops=dict(marker="D", markersize=7, markeredgecolor="black", markerfacecolor="white"),
#         medianprops=dict(color="black", linewidth=2),
#         whiskerprops=dict(color="gray"),
#         capprops=dict(color="gray"),
#         boxprops=dict(color="gray")
#     )

#     # overlay explicit mean/median markers for legend clarity
#     x = np.arange(1, len(data) + 1)
#     means = [np.mean(d) for d in data]
#     meds  = [np.median(d) for d in data]

#     mean_sc = ax.scatter(x, means, marker="D", s=60, facecolors="white", edgecolors="black", zorder=3, label="Mean")
#     med_sc  = ax.scatter(x, meds,  marker="s", s=45, color="black", zorder=3, label="Median")

#     # write values above each box
#     for xi, m, md in zip(x, means, meds):
#         ax.text(xi, m, f"{m:.1f}", ha="center", va="bottom", fontsize=9, color="blue")
#         ax.text(xi, md, f"{md:.1f}", ha="center", va="top", fontsize=9, color="red")

#     ax.set_ylabel("Angle (deg)")
#     ax.legend(handles=[mean_sc, med_sc], loc="best", frameon=False)

#     plt.tight_layout()
#     plt.savefig(OUT_DIR / "MI_combined_whiskers_angles.svg")
#     plt.close(fig)

    

# dists = [by_group[k]["distances"] for k in ["subs_pattern_A", "subs_pattern_B", "subs_weak"] if by_group[k]["distances"]]
# if dists:
#     plt.boxplot(dists, labels=["A", "B", "Weak"], showmeans=True)
#     plt.ylabel("Euclidean distance")
#     plt.savefig(OUT_DIR / "MI_combined_whiskers_distances.svg")
#     plt.close()

# ###################################################################################################### Plotting rest STFTs
# # ── Rest Plots ─────────────────────────────────────────
# def load_pkl_group(base_path):
#     with open(f"{base_path}\\data_dict.pkl", "rb") as f: data = pickle.load(f)
#     with open(f"{base_path}\\Pattern_A.pkl", "rb") as f: A = pickle.load(f)
#     with open(f"{base_path}\\Pattern_B.pkl", "rb") as f: B = pickle.load(f)
#     with open(f"{base_path}\\Weak.pkl", "rb") as f: W = pickle.load(f)
#     return data, A, B, W

# data1, A1, B1, W1 = load_pkl_group(path1)
# data2, A2, B2, W2 = load_pkl_group(path2)
# data1 = {k: v * 1e-3 for k, v in data1.items()}
# combined_data = {**data1, **data2}
# groups = {"Pattern_A": list(A1) + list(A2), "Pattern_B": list(B1) + list(B2), "Weak": list(W1) + list(W2)}

# # Set global font size scaling (double the default)
# plt.rcParams.update({'font.size': 24})  # Default is ~12, so this doubles everything

# for group_name, ids in groups.items():
#     mats = [combined_data[sid] for sid in ids if sid in combined_data]
#     if not mats: continue
#     avg_map = np.nanmean(np.stack(mats, axis=0), axis=0)
#     colors = [(0, 'blue'), (0.3, 'skyblue'), (0.5, 'black'), (0.7, 'lightyellow'), (1, 'yellow')]
#     cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

#     plt.figure(figsize=(12, 9))  # Slightly larger figure for readability
#     im = plt.imshow(avg_map, aspect='auto', cmap=cmap, interpolation='nearest',
#                     vmin=-2e-6, vmax=2e-6)

#     cbar = plt.colorbar(im)
#     cbar.set_ticks([-2e-6, 0, 2e-6])
#     cbar.set_ticklabels(["-2e-6", "0", "2e-6"])
#     cbar.set_label("Amplitude [V]", rotation=270, labelpad=25, fontsize=26)

#     tick_labels = [0, 15, 30, 45, 60]
#     tick_pos = [int(i * (avg_map.shape[1] / 60)) for i in tick_labels]
#     plt.xticks(tick_pos, [str(i) for i in tick_labels], fontsize=24)
#     plt.yticks(fontsize=24)

#     plt.xlabel("Time [s]", fontsize=28)
#     plt.ylabel("Frequency [Hz]", fontsize=28)

#     plt.ylim(0, 20)
#     plt.tight_layout()
#     plt.savefig(OUT_DIR / f"Rest_avg_{group_name}_Combined.svg")
#     plt.close()

# ###################################################################################################### ERS window plotting
# ORIG_LEN = 1500
# PAD = 250
# TARGET_LEN = 1280
# PATTERN = "groups*.pkl"

# WEIGHTS = {"A": (2, 12), "B": (4, 7), "W": (12, 84)}

# # sample-index markers (on 1280-sample timeline)
# DARK_SPAN = (560, 720)   # shaded region [start_idx, end_idx)
# VLINE_DS = 640           # vertical line index (0 s)

# AVG_WIN = 20  # average every 20 consecutive samples (block-avg)

# # ---------- helpers ----------
# def _find_latest_pickle(folder, pattern=PATTERN):
#     cands = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getmtime)
#     if not cands:
#         raise FileNotFoundError(f"No pickle found in {folder} matching {pattern}")
#     return cands[-1]

# def _avg_cat(baseline_list, mi_list):
#     b = np.mean(np.asarray(baseline_list), axis=0)
#     m = np.mean(np.asarray(mi_list), axis=0)
#     return np.concatenate([b, m], axis=0)

# def _series_from_groups(groups):
#     out = {}
#     for lbl, (c3_base, c3_mi, c4_base, c4_mi) in groups.items():
#         ll = _avg_cat(c3_base, c3_mi)
#         rl = _avg_cat(c4_base, c4_mi)
#         out[lbl] = (ll, rl)
#     return out

# def _minmax_scale(arr):
#     arr = np.asarray(arr, dtype=float)
#     amin, amax = np.min(arr), np.max(arr)
#     if amax - amin == 0:
#         return np.zeros_like(arr)
#     return (arr - amin) / (amax - amin)

# def _pad_if_orig_len(arr, orig_len=ORIG_LEN, pad=PAD):
#     if len(arr) == orig_len:
#         return np.concatenate([np.zeros(pad), arr, np.zeros(pad)])
#     return arr

# def _resample_to_len(arr, target_len):
#     n = len(arr)
#     if n == target_len:
#         return arr.copy()
#     x_old = np.linspace(0.0, 1.0, n)
#     x_new = np.linspace(0.0, 1.0, target_len)
#     return np.interp(x_new, x_old, arr)

# def _align_and_weighted_avg(a, b, w1, w2):
#     n = min(len(a), len(b))
#     return (w1 * a[:n] + w2 * b[:n]) / (w1 + w2)

# def _block_avg_with_time(y, block_size=AVG_WIN, t_start=-4.0, t_end=4.0, n_full=TARGET_LEN):
#     """Return block-averaged signal and corresponding block-centered time axis."""
#     t = np.linspace(t_start, t_end, n_full, endpoint=False)   # full-resolution time
#     y = y[:n_full]
#     n_trim = n_full - (n_full % block_size)                   # 1280 divisible by 20
#     yb = y[:n_trim].reshape(-1, block_size).mean(axis=1)
#     tb = t[:n_trim].reshape(-1, block_size).mean(axis=1)
#     return tb, yb

# def _smooth(y, k=5):
#     """Simple moving-average smoothing (k odd recommended)."""
#     k = max(1, int(k))
#     kernel = np.ones(k) / k
#     return np.convolve(y, kernel, mode="same")

# # ---------- main ----------
# def combined_ers_from_pickles(dir1, dir2, pattern=PATTERN, weights=WEIGHTS,
#                               do_scale=True, save_svg=True):

#     # ~2x larger fonts
#     plt.rcParams.update({"font.size": 24})

#     pkl1 = _find_latest_pickle(dir1, pattern)
#     pkl2 = _find_latest_pickle(dir2, pattern)

#     with open(pkl1, "rb") as f: g1 = pickle.load(f)
#     with open(pkl2, "rb") as f: g2 = pickle.load(f)

#     s1 = _series_from_groups(g1)
#     s2 = _series_from_groups(g2)

#     common = sorted(set(s1.keys()) & set(s2.keys()) & set(weights.keys()))
#     if not common:
#         raise KeyError("No common labels between the two pickles (or weights).")

#     # convert shaded span indices to block indices (for later → to time)
#     b0, b1 = DARK_SPAN[0] // AVG_WIN, DARK_SPAN[1] // AVG_WIN
#     vline_time = 0.0  # 0 s vertical line

#     for lbl in common:
#         ll1, rl1 = s1[lbl]
#         ll2, rl2 = s2[lbl]

#         if do_scale:
#             ll1, rl1 = _minmax_scale(ll1), _minmax_scale(rl1)
#             ll2, rl2 = _minmax_scale(ll2), _minmax_scale(rl2)

#         # dataset1: pad then resample
#         ll1 = _resample_to_len(_pad_if_orig_len(ll1), TARGET_LEN)
#         rl1 = _resample_to_len(_pad_if_orig_len(rl1), TARGET_LEN)

#         # dataset2: resample
#         ll2 = _resample_to_len(ll2, TARGET_LEN)
#         rl2 = _resample_to_len(rl2, TARGET_LEN)

#         # weighted average
#         w1, w2 = weights[lbl]
#         ll = _align_and_weighted_avg(ll1, ll2, w1, w2)
#         rl = _align_and_weighted_avg(rl1, rl2, w1, w2)

#         # block-average + time axis
#         t_ll, ll_b = _block_avg_with_time(ll, block_size=AVG_WIN)
#         t_rl, rl_b = _block_avg_with_time(rl, block_size=AVG_WIN)

#         # smooth
#         ll_s = _smooth(ll_b, k=5)
#         rl_s = _smooth(rl_b, k=5)

#         # shaded span as times
#         span_t0, span_t1 = t_ll[b0], t_ll[b1-1]

#         # plot
#         fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 9))
#         for ax, t, y, limb in zip(
#             axes,
#             [t_ll, t_rl],
#             [ll_s, rl_s],
#             ["Left Limb (LL)", "Right Limb (RL)"]
#         ):
#             ax.plot(t, y, linewidth=2)                 # smooth curve
#             ax.axvline(x=vline_time, linestyle="--", color="k")
#             ax.axvspan(span_t0, span_t1, color="black", alpha=0.5)
#             ax.set_ylabel("Normalized Power")
#             ax.set_xlim(-4.0, 4.0)
#             ax.tick_params(labelbottom=True)           # ensure x ticks visible on both

#         # show x-axis label on both (since sharex hides top by default)
#         for ax in axes:
#             ax.set_xlabel("Time [s]")

#         plt.tight_layout()
#         outpath = f"{lbl}_erss_weighted_avg_block{AVG_WIN}.svg"
#         if save_svg:
#             plt.savefig(outpath, format="svg")
#             print(f"Saved {outpath}")
#         plt.close(fig)
# # call
# combined_ers_from_pickles(path1, path2, pattern=PATTERN)

# ###################################################################################################### plotting baseline power whiskers (one dataset)

# def plot_whiskers_from_csv(
#     csv_path,
#     prefix,                 # "mi_p" or "baseline_p"
#     out_path=None,
#     ylabel=None,
#     show_mean_text=True,
#     show_pair_deltas=True,  # add Δ between (1–2) and (3–4)
#     mean_scale=1.0,         # e.g., 1e18 if you want to scale values
#     figsize=(10, 6),
#     dpi=300
# ):
#     """Plot box/whisker for a given prefix and save the figure.
#        Whisker order: c4_rl_A, c4_rl_weak, c3_ll_B, c3_ll_weak
#     """
#     suffixes = ["c4_rl_A", "c4_rl_weak", "c3_ll_B", "c3_ll_weak"]
#     labels   = ["C4 RL A", "C4 RL Weak", "C3 LL B", "C3 LL Weak"]
#     cols     = [f"{prefix}_{s}" for s in suffixes]

#     df = pd.read_csv(csv_path)
#     keep_cols = [c for c in cols if c in df.columns]
#     if not keep_cols:
#         raise ValueError(f"No columns with prefix '{prefix}' found in {csv_path}.")

#     data = [df[c].dropna().values for c in keep_cols]
#     keep_labels = [labels[cols.index(c)] for c in keep_cols]

#     fig, ax = plt.subplots(figsize=figsize)
#     bp = ax.boxplot(data, labels=keep_labels, patch_artist=True)

#     # Means (scaled) for annotations + deltas
#     means = [float(np.mean(d)) if len(d) else np.nan for d in data]
#     means_scaled = [m * mean_scale if np.isfinite(m) else np.nan for m in means]

#     # Per-whisker mean labels
#     if show_mean_text:
#         for i, (d, m) in enumerate(zip(data, means_scaled), start=1):
#             if len(d) == 0 or not np.isfinite(m):
#                 continue
#             y_pos = np.max(d) * 1.02
#             ax.text(i, y_pos, f"Mean={m:}", ha="center", va="bottom", fontsize=9, color="blue")

#     # Pairwise Δ annotations: (1–2) and (3–4)
#     if show_pair_deltas and len(data) >= 4:
#         # pick y positions above each pair
#         y12 = max(np.max(data[0]) if len(data[0]) else 0,
#                   np.max(data[1]) if len(data[1]) else 0) * 1.12
#         y34 = max(np.max(data[2]) if len(data[2]) else 0,
#                   np.max(data[3]) if len(data[3]) else 0) * 1.12

#         # compute deltas using the same scaling
#         if np.isfinite(means_scaled[0]) and np.isfinite(means_scaled[1]):
#             delta12 = means_scaled[0] - means_scaled[1]
#             ax.text(1.5, y12, f"Δ(1–2)={delta12}", ha="center", va="bottom", fontsize=10)

#         if np.isfinite(means_scaled[2]) and np.isfinite(means_scaled[3]):
#             delta34 = means_scaled[2] - means_scaled[3]
#             ax.text(3.5, y34, f"Δ(3–4)={delta34}", ha="center", va="bottom", fontsize=10)

#     ax.set_ylabel(ylabel if ylabel else prefix.replace("_", " ").upper())
#     ax.set_xlabel("Channels / Conditions")
#     plt.tight_layout()

#     if out_path is None:
#         base = os.path.splitext(os.path.basename(csv_path))[0]
#         out_path = f"{base}_{prefix}.png"

#     plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
#     plt.close(fig)

#     # return paths and raw means (unscaled) + deltas (scaled, for logging)
#     result = {
#         "means_raw": dict(zip(keep_labels, means)),
#         "means_scaled": dict(zip(keep_labels, means_scaled)),
#     }
#     if len(data) >= 4:
#         result["delta_1_2_scaled"] = (means_scaled[0] - means_scaled[1]
#                                       if np.isfinite(means_scaled[0]) and np.isfinite(means_scaled[1]) else np.nan)
#         result["delta_3_4_scaled"] = (means_scaled[2] - means_scaled[3]
#                                       if np.isfinite(means_scaled[2]) and np.isfinite(means_scaled[3]) else np.nan)

#     return out_path, result



# csv_path_mi   = os.path.join(path2, "mi_p_values_dataset1.csv")
# plot_whiskers_from_csv(csv_path_mi,   prefix="mi_p",       out_path="mi_p_1.png",       ylabel="MI P")

# csv_path_base = os.path.join(path2, "baseline_p_values_dataset1.csv")
# plot_whiskers_from_csv(csv_path_base, prefix="baseline_p", out_path="baseline_p_1.png", ylabel="Baseline P")

# csv_path_mi   = os.path.join(path1, "mi_p_values_dataset2.csv")
# plot_whiskers_from_csv(csv_path_mi,   prefix="mi_p",       out_path="mi_p_2.png",       ylabel="MI P")

# csv_path_base = os.path.join(path1, "baseline_p_values_dataset2.csv")
# plot_whiskers_from_csv(csv_path_base, prefix="baseline_p", out_path="baseline_p_2.png", ylabel="Baseline P")


# # if ERS instead of P, results are slightly better.
stop=1