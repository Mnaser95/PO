import mne
from scipy.signal import stft
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import pandas as pd
from scipy.stats import pearsonr
from collections import Counter
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy.signal import welch
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_1samp
warnings.filterwarnings("ignore")
from scipy.stats import kruskal
from scipy.stats import f_oneway
from pathlib import Path
import csv
import pickle

N_SUBJECT = 109
BASELINE_EYE_CLOSED = [2]
IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]
SELECTED_CHANNELS = [8,9,1,2,15,16, 11,12,4,5,18,19]
num_chan_per_hemi=len(SELECTED_CHANNELS)//2
fs=160
num_runs_sub=3
needed_subs=109
low_rest=1
high_rest=20
low_MI=8
high_MI=12
needed_ratio=1.4
MI_essence = "PSD"
generate_rest_maps = False
generate_MI_plots = False

#################################################################################################################################################### Clustering
def raw_rest_processing(raw):
    raw.filter(l_freq=low_rest, h_freq=high_rest, picks="eeg", verbose='WARNING')
    events, _ = mne.events_from_annotations(raw)

    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    df = pd.read_csv(fr"64montage.csv", header=None, names=["name", "x", "y", "z"])    
    ch_pos = {row['name']: [row['x'], row['y'], row['z']] for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage,on_missing="warn")
    valid_chs = [
    ch['ch_name'] for ch in raw.info['chs']
    if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()
    ]
    raw= raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)
    #raw.plot_sensors(show_names=True)  # just to verify positions

    epoched = mne.Epochs(raw,events,event_id=dict(rest=1),tmin=1,tmax=59,proj=False,picks=SELECTED_CHANNELS,baseline=None,preload=True)

    return epoched
def rest_data_generation():
    X = (epoched.get_data() * 1e3).astype(np.float32)

    avg_left = X[:, :num_chan_per_hemi, :].mean(axis=1)  # (n_samples, time)
    avg_right = X[:, -num_chan_per_hemi:, :].mean(axis=1)

    # Apply STFT per sample and average across samples
    Zxx1_total = []
    Zxx2_total = []
    f, _, Z1 = stft(avg_left[0], fs, nperseg=fs)
    f, _, Z2 = stft(avg_right[0], fs, nperseg=fs)
    Zxx1_total.append(np.abs(Z1))
    Zxx2_total.append(np.abs(Z2))

    Zxx1_mean = np.mean(Zxx1_total, axis=0)
    Zxx2_mean = np.mean(Zxx2_total, axis=0)
    data = (Zxx1_mean - Zxx2_mean).squeeze()
    data_diff = (Zxx1_mean**2 - Zxx2_mean**2).squeeze()

    return data, data_diff
def subject_select(mid,other):
    segment_size = 10
    n_segments = len(mid) // segment_size
    votes=[]
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        avg_mid_abs = np.abs(np.mean(mid[start:end]))
        avg_other_abs = np.abs(np.mean(other[start:end]))
        ratio = avg_mid_abs / avg_other_abs
        if ratio > needed_ratio:
            if np.mean(mid[start:end])>0:
                votes.append("11") 
            else:
                votes.append("10")
        else:
            votes.append("0X")

    #"11": strong, positive (mainly yellow)
    #"10": strong, negative (mainly blue)
    #"0X": weak 

    vote_counts = Counter(votes)
    majority_vote, num_majority_votes = vote_counts.most_common(1)[0]

    if majority_vote=="11":
        res="Pattern B"
    if majority_vote=="10":
        res="Pattern A"
    if majority_vote=="0X":
        res="Weak"
    return res
def rest_plotting():
    # Define a custom colormap
    colors = [
        (0, 'blue'),
        (0.3, 'skyblue'),
        (0.5, 'black'),
        (0.7, 'lightyellow'),
        (1, 'yellow')
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        data,
        aspect='auto',
        cmap=custom_cmap,
        interpolation='nearest',
        vmin=-5e-6,
        vmax=5e-6
    )

    # Custom colorbar
    cbar = plt.colorbar(im)
    cbar.set_ticks([-5e-6, 0, 5e-6])
    cbar.set_ticklabels(["Min (-ve)", "0", "Max (+ve)"])
    cbar.ax.tick_params(labelsize=12)

    # Custom x-axis ticks (e.g., 5 ticks between 0 and 60)
    tick_labels = [0, 15, 30, 45, 60]
    tick_positions = [int(i * (data.shape[1] / 60)) for i in tick_labels]
    plt.xticks(tick_positions, [str(i) for i in tick_labels], fontsize=12)
    plt.yticks(fontsize=12)

    # Axis labels
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("Frequency [Hz]", fontsize=14)

    # Optional: Add a title
    plt.title(fr"Rest: |STFT| left hemi - |STFT| right hemi", fontsize=16)

    plt.ylim(0, 20)
    plt.tight_layout()
    out_path =Path(OUT_DIR) / f"Rest_{sub+1:03d}.svg"
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close()
def plotting_rest_maps_all_subs(data_dict):
    def _avg_group(ids):
        mats = [np.asarray(data_dict[sid]) for sid in ids if sid in data_dict and data_dict[sid] is not None]
        miss = [sid for sid in ids if sid not in data_dict or data_dict[sid] is None]
        if not mats:
            return None, miss
        shapes = {m.shape for m in mats}
        if len(shapes) != 1:
            raise ValueError(f"Arrays must share shape. Got: {shapes}")
        return np.nanmean(np.stack(mats, axis=0), axis=0), miss

    def _plot_and_save(avg_map, group_name, used, missing):
        actual_min = np.nanmin(avg_map)
        actual_max = np.nanmax(avg_map)

        colors = [(0, 'blue'), (0.3, 'skyblue'), (0.5, 'black'),
                  (0.7, 'lightyellow'), (1, 'yellow')]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(avg_map, aspect='auto', cmap=cmap,
                        interpolation='nearest', vmin=-5e-6, vmax=5e-6)
        cbar = plt.colorbar(im)
        cbar.set_ticks([-5e-6, actual_min, 0, actual_max, 5e-6])
        cbar.set_ticklabels([
            f"Fixed Min\n({-5e-6:.1e})",
            f"Data Min\n({actual_min:.1e})",
            "0",
            f"Data Max\n({actual_max:.1e})",
            f"Fixed Max\n({5e-6:.1e})"
        ])
        cbar.ax.tick_params(labelsize=10)

        tick_labels = [0, 15, 30, 45, 60]
        tick_pos = [int(i * (avg_map.shape[1] / 60)) for i in tick_labels]
        plt.xticks(tick_pos, [str(i) for i in tick_labels], fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel("Time [s]", fontsize=18)
        plt.ylabel("Frequency [Hz]", fontsize=18)
        title = f"Rest: |STFT| LH - RH • {group_name} • n={len(used)}"
        if missing:
            title += f" (missing: {len(missing)})"
        plt.title(title, fontsize=16)

        plt.ylim(0, 20)
        plt.tight_layout()
        plt.savefig(f"Rest_avg_{group_name}.svg")
        plt.close()

    groups = {
        "Pattern_A": list(subs_pattern_A),
        "Pattern_B": list(subs_pattern_B),
        "Weak": list(subs_weak),
    }

    results = {}
    for gname, ids in groups.items():
        avg_map, missing = _avg_group(ids)
        if avg_map is None:
            results[gname] = {
                "used_ids": [],
                "missing_ids": missing,
                "path": None,
                "avg": None
            }
            continue
        used = [sid for sid in ids if sid not in missing]
        actual_min = np.nanmin(avg_map)
        actual_max = np.nanmax(avg_map)

        _plot_and_save(avg_map, gname, used, missing)
        results[gname] = {
            "used_ids": used,
            "missing_ids": missing,
            "path": f"Rest_avg_{gname}.svg",
            "avg": avg_map,
            "actual_min": actual_min,
            "actual_max": actual_max
        }

    return results

def export_rest_data(data_dict):
    # Save each variable as a separate pickle file
    with open('data_dict.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    
    with open('Pattern_A.pkl', 'wb') as f:
        pickle.dump(subs_pattern_A, f)
    
    with open('Pattern_B.pkl', 'wb') as f:
        pickle.dump(subs_pattern_B, f)
    
    with open('Weak.pkl', 'wb') as f:
        pickle.dump(subs_weak, f)

# Load data
physionet_paths = [mne.datasets.eegbci.load_data(subject_id,BASELINE_EYE_CLOSED,"/root/mne_data" ) for subject_id in range(1, needed_subs + 1) ]
physionet_paths = np.concatenate(physionet_paths)
parts = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING')for path in physionet_paths] # 0-indexed

# Create directory to save figures
OUT_DIR = Path("Rest Maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

patterns={}
data_all_subs={}
for sub, raw in enumerate(parts):
    epoched=raw_rest_processing(raw)

    data, data_p=rest_data_generation() # multiplied by 1000

    data_all_subs[sub+1]=data

    avg_mid_freq=data[7:13,:].mean(axis=0)
    avg_below_freq=data[2:7,:].mean(axis=0)

    res_down=subject_select(avg_mid_freq,avg_below_freq)

    patterns[sub+1] = res_down

    if generate_rest_maps: rest_plotting()

# for k in [k for k, v in patterns.items() if v == "Weak"]: #remove weak
#     del patterns[k]
#     del confidences[k]

for k in [38,88,89,92,100,104]: #remove incorrecrt datapoints 
    if k in patterns:
        del patterns[k]

# all patterns exist now, except for 6 wrong subjects. All starts from 1 not 0 (total left is 103)
subs_taken=list(patterns.keys())
subs_pattern_B = [k for k, v in patterns.items() if v == "Pattern B"]
subs_pattern_A = [k for k, v in patterns.items() if v == "Pattern A"]
subs_weak = [k for k, v in patterns.items() if v == "Weak"]

plotting_rest_maps_all_subs(data_all_subs)
export_rest_data(data_all_subs)

#################################################################################################################################################### Next
def raw_MI_processing(raw):
    raw.filter(l_freq=low_MI, h_freq=high_MI, picks="eeg", verbose='WARNING')
    
    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    
    df = pd.read_csv(fr"64montage.csv", header=None, names=["name", "x", "y", "z"])    
    ch_pos = {row['name']: [row['x'], row['y'], row['z']] for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage,on_missing="warn")
    valid_chs = [
    ch['ch_name'] for ch in raw.info['chs']
    if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()
    ]
    raw= raw.copy().pick_channels(valid_chs)
    raw = mne.preprocessing.compute_current_source_density(raw)

    events, _ = mne.events_from_annotations(raw)

    epoched = mne.Epochs(raw, events, event_id=dict(left=2, right=3), tmin=0, tmax=4,
                        proj=False, picks=SELECTED_CHANNELS, baseline=None, preload=True)
    baseline_epoched = mne.Epochs(raw, events, event_id=dict(base=1), tmin=0, tmax=4,
                                proj=False, picks=SELECTED_CHANNELS, baseline=None, preload=True)

    # Sample positions
    mi_samples = epoched.events[:, 0]         # retained MI event samples
    base_samples = baseline_epoched.events[:, 0]

    # Keep baselines with an MI event within 1000 samples
    valid_baseline_mask = np.array([
        np.any((mi_samples - base_sample > 0) & (mi_samples - base_sample <= 1000))
        for base_sample in base_samples
    ])
    filtered_baseline = baseline_epoched[valid_baseline_mask]

    return epoched,epoched.get_data(),filtered_baseline,filtered_baseline.get_data()
def MI_data_generation(epoched):
    X = (epoched.get_data() * 1e3).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)

    _, _, Zxx = stft(X, fs, nperseg=fs)
    MI_tf = np.abs(Zxx)
    X_tf = MI_tf.mean(axis=2).mean(axis=2) # this will avg over time and freq

    f, psd_run = welch(X, fs=fs)
    psd_run_avg_freq=psd_run.mean(axis=2)
    psd_run_avg_freq_left = psd_run_avg_freq[:, :num_chan_per_hemi].mean(axis=1)
    psd_run_avg_freq_right = psd_run_avg_freq[:, -num_chan_per_hemi:].mean(axis=1)

    avg_left = X_tf[:, :num_chan_per_hemi].mean(axis=1)
    avg_right = X_tf[:, -num_chan_per_hemi:].mean(axis=1)
    return avg_left,avg_right,y,psd_run_avg_freq_left,psd_run_avg_freq_right
def plotting_MI_datapoints(data1,data2,labels_MI):
    
    unique_labels = np.unique(labels_MI)

    # Prepare scatter plot for individual points
    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        label_text = "LH-MI" if label == 0 else "RH-MI"
        color = plt.cm.viridis((label) / (len(unique_labels) - 1))

        # Plot individual points
        plt.scatter(
            data1[labels_MI == label],
            data2[labels_MI == label],
            c=[color],
            label=label_text,
            s=100
        )

        # Plot average point as a large 'X'
        avg_x = np.mean(data1[labels_MI == label])
        avg_y = np.mean(data2[labels_MI == label])
        plt.scatter(
            avg_x,
            avg_y,
            c=[color],
            s=200,
            marker='X',
            edgecolors='black',
            linewidths=1.5,
            label=f"{label_text} (avg)"
        )

    # Remove tick marks but keep axis labels
    plt.tick_params(axis='both', which='both', length=0)
    plt.xticks([])
    plt.yticks([])

    # Labels and title
    plt.xlabel(r"$\mathrm{PSD}_{\mathrm{MI,\ LH}}$", fontsize=14)
    plt.ylabel(r"$\mathrm{PSD}_{\mathrm{MI,\ RH}}$", fontsize=14)
    plt.title("MI Distribution", fontsize=16)

    # Legend and grid
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    out_path = OUT_DIR / f"MI_{sub_mod}.svg"
    plt.savefig(out_path, format="svg")
    plt.clf()

def find_angs_dist(data1_dict, data2_dict, labels_dict):
    groups = {
        "subs_pattern_A": subs_pattern_A,
        "subs_pattern_B": subs_pattern_B,
        "subs_weak": subs_weak,
    }
    out = {k: {"pairs": [], "angles": [], "distances": [], "rows": []} for k in groups.keys()}
    for gkey, subject_list in groups.items():
        for sid in subject_list:
            if sid not in data1_dict or sid not in data2_dict or sid not in labels_dict:
                continue
            x = np.asarray(data1_dict[sid]); y = np.asarray(data2_dict[sid]); lb = np.asarray(labels_dict[sid])
            n = min(len(x), len(y), len(lb))
            if n == 0: continue
            x, y, lb = x[:n], y[:n], lb[:n]

            pts = {}
            for lbl in (0, 1):  # 0=LH, 1=RH
                m = (lb == lbl)
                if np.any(m):
                    pts[lbl] = (x[m].mean(), y[m].mean())

            if 0 in pts and 1 in pts:
                (x0, y0), (x1, y1) = pts[0], pts[1]
                dx, dy = x1 - x0, y1 - y0
                theta = np.degrees(np.arctan2(dy, dx));  theta = theta if theta >= 0 else theta + 360
                dist = float(np.hypot(dx, dy))

                out[gkey]["pairs"].append((x0, y0, x1, y1))
                out[gkey]["angles"].append(theta)
                out[gkey]["distances"].append(dist)
                out[gkey]["rows"].append({
                    "group": gkey, "subject": sid,
                    "x_LH": x0, "y_LH": y0, "x_RH": x1, "y_RH": y1,
                    "angle_deg": theta, "distance": dist, "dataset": "set1"
                })
    return out
def restructuring(by_group):
    groups = ("subs_pattern_A", "subs_pattern_B", "subs_weak")

    out = {}
    for gkey in groups:
        angles, dists = [], []
        for (x0, y0, x1, y1) in by_group[gkey]["pairs"]:
            dx, dy = x1 - x0, y1 - y0
            theta = float(np.degrees(np.arctan2(dy, dx)))
            if theta < 0:
                theta += 360.0
            angles.append(theta)
            dists.append(float(np.hypot(dx, dy)))

        if angles:
            ang_rad = np.radians(angles)
            z = np.mean(np.exp(1j * ang_rad))
            R = float(np.abs(z))
            mean_angle = float((np.degrees(np.angle(z)) % 360))
            mean_dist = float(np.mean(dists))
        else:
            R, mean_angle, mean_dist = 0.0, np.nan, np.nan

        out[gkey] = {
            "pairs": by_group[gkey]["pairs"],
            "angles_deg": angles,
            "distances": dists,
            "circular_mean_angle_deg": mean_angle,
            "resultant_length": R,
            "mean_distance": mean_dist,
            "n": len(angles),
        }
    return out
def plotting_MI_datapoints_all(results, label_names=None, colors=None):
    groups = {
        "subs_pattern_A": subs_pattern_A,
        "subs_pattern_B": subs_pattern_B,
        "subs_weak": subs_weak,
    }
    label_names = {0: "LH-MI", 1: "RH-MI"}
    colors = {0: plt.cm.viridis(0.0), 1: plt.cm.viridis(1.0)}


    # unified limits
    all_x, all_y = [], []
    for gkey, info in results.items():
        for (x0, y0, x1, y1) in info["pairs"]:
            all_x += [x0, x1]; all_y += [y0, y1]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    mx = 0.05 * (x_max - x_min) if x_max > x_min else 1
    my = 0.05 * (y_max - y_min) if y_max > y_min else 1
    x_min, x_max = x_min - mx, x_max + mx
    y_min, y_max = y_min - my, y_max + my

    for gkey, info in results.items():
        plt.figure(figsize=(8, 6))
        plotted_lbl = set()

        for (x0, y0, x1, y1) in info["pairs"]:
            plt.scatter(x0, y0, c=[colors[0]], s=200, marker='*',
                        edgecolors='black', linewidths=1.0,
                        label=label_names[0] if 0 not in plotted_lbl else None)
            plt.scatter(x1, y1, c=[colors[1]], s=200, marker='*',
                        edgecolors='black', linewidths=1.0,
                        label=label_names[1] if 1 not in plotted_lbl else None)
            plotted_lbl.update({0, 1})
            plt.plot([x0, x1], [y0, y1], color="gray", linewidth=1, alpha=0.7)

            # annotate angle
            dx, dy = x1 - x0, y1 - y0
            theta = np.degrees(np.arctan2(dy, dx))
            if theta < 0: theta += 360
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            plt.text(xm, ym, f"{theta:.1f}°", fontsize=9, color="gray",
                     ha="center", va="center", alpha=0.8)

        plt.tick_params(axis='both', which='both', length=0)
        plt.xticks([]); plt.yticks([])
        plt.xlabel(r"$\mathrm{PSD}_{\mathrm{MI,\ LH}}$", fontsize=14)
        plt.ylabel(r"$\mathrm{PSD}_{\mathrm{MI,\ RH}}$", fontsize=14)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        if plotted_lbl:
            plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        # summary box
        if info["n"] > 0:
            txt = f"n={info['n']}\nmean θ={info['circular_mean_angle_deg']:.1f}°\nmean d={info['mean_distance']:.4f}"
            ax = plt.gca()
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9),
                    fontsize=12)

        plt.savefig(f"MI_{gkey}.svg")
        plt.close()
def plot_MI_whiskers(by_group):
    order = ["subs_pattern_A", "subs_pattern_B", "subs_weak"]
    names = ["A", "B", "Weak"]
    out_dir = Path(globals().get("OUT_DIR", "."))

    angle_data = [by_group[k]["angles"] for k in order if by_group[k]["angles"]]
    angle_labels = [n for k,n in zip(order, names) if by_group[k]["angles"]]
    if angle_data:
        plt.figure(figsize=(6, 5))
        plt.boxplot(angle_data, labels=angle_labels, showmeans=True)
        plt.ylabel("Angle (deg)"); plt.tight_layout()
        plt.savefig(fr"MI_whiskers_angles.svg"); plt.clf()

    dist_data = [by_group[k]["distances"] for k in order if by_group[k]["distances"]]
    dist_labels = [n for k,n in zip(order, names) if by_group[k]["distances"]]
    if dist_data:
        plt.figure(figsize=(6, 5))
        plt.boxplot(dist_data, labels=dist_labels, showmeans=True)
        plt.ylabel("Euclidean distance"); plt.tight_layout()
        plt.savefig(fr"MI_whiskers_distances.svg"); plt.clf()
def export_mi_data_csv(by_group):
    out_file = fr"MI_data.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group","subject","x_LH","y_LH","x_RH","y_RH","angle_deg","distance","dataset"])
        for gk, d in by_group.items():
            for r in d.get("rows", []):
                w.writerow([r["group"], r["subject"], r["x_LH"], r["y_LH"],
                            r["x_RH"], r["y_RH"], r["angle_deg"], r["distance"], r["dataset"]])
    print(f"Saved: {out_file}")

def ers_process(ll_lbl, rl_lbl):
    for case, lbl_id in (("ll", ll_lbl), ("rl", rl_lbl)):

        mi_mask = labels_all == lbl_id
        mi_full = epochses[mi_mask]      # (n_epochs_MI, n_channels, n_timepoints)
        mi = np.stack(mi_full, axis=0)
        power_c3_over_epochs_and_time = mi[:, :6, :].mean(axis=1)**2   # (n_epochs, n_time)
        power_c4_over_epochs_and_time = mi[:, 6:12, :].mean(axis=1)**2
        power_c3_over_time = power_c3_over_epochs_and_time.mean(axis=0)  # (n_time,)
        power_c4_over_time = power_c4_over_epochs_and_time.mean(axis=0)

        baseline_full = epochses_baseline[mi_mask]
        mi_base = np.stack(baseline_full, axis=0)
        power_c3_over_epochs_and_time_baseline = mi_base[:, :6, :].mean(axis=1)**2   # (n_epochs, n_time)
        power_c4_over_epochs_and_time_baseline = mi_base[:, 6:12, :].mean(axis=1)**2
        power_c3_over_time_baseline = power_c3_over_epochs_and_time_baseline.mean(axis=0)  # (n_time,)
        power_c4_over_time_baseline = power_c4_over_epochs_and_time_baseline.mean(axis=0)



        if sub_mod in subs_pattern_A:
            if case == "ll":
                power_c3_over_time_A.append(power_c3_over_time)
                power_c3_over_time_baseline_A.append(power_c3_over_time_baseline)
            else:
                power_c4_over_time_A.append(power_c4_over_time)
                power_c4_over_time_baseline_A.append(power_c4_over_time_baseline)
        elif sub_mod in subs_pattern_B:
            if case == "ll":
                power_c3_over_time_B.append(power_c3_over_time)
                power_c3_over_time_baseline_B.append(power_c3_over_time_baseline)
            else:
                power_c4_over_time_B.append(power_c4_over_time)
                power_c4_over_time_baseline_B.append(power_c4_over_time_baseline)
        elif sub_mod in subs_weak:
            if case == "ll":
                power_c3_over_time_W.append(power_c3_over_time)
                power_c3_over_time_baseline_W.append(power_c3_over_time_baseline)
            else:
                power_c4_over_time_W.append(power_c4_over_time)
                power_c4_over_time_baseline_W.append(power_c4_over_time_baseline)



        step=160
        i_1=80 # start from this many seconds after MI cue
        i_2=i_1+step//2
        ers_c3_max=-10000
        ers_c4_max=-10000

        while i_2 <= 640: 
            a = ((power_c3_over_time[i_1:i_2].mean() - power_c3_over_time_baseline[:560].mean())
                    / power_c3_over_time_baseline[:560].mean()) * 100
            b = ((power_c4_over_time[i_1:i_2].mean() - power_c4_over_time_baseline[:560].mean())
                    / power_c4_over_time_baseline[:560].mean()) * 100
            
            ers_c3_max=max(a,ers_c3_max)
            ers_c4_max=max(b,ers_c4_max)

            i_1+=step//2
            i_2+=step//2

        i1, i2 = 0, 560
        e_baseline_c3 = np.sum(power_c3_over_time_baseline[i1:i2+1]) * (1/160)
        e_baseline_c4 = np.sum(power_c4_over_time_baseline[i1:i2+1]) * (1/160)
        i1, i2 = 80, 640
        e_c3 = np.sum(power_c3_over_time[i1:i2+1]) * (1/160)
        e_c4 = np.sum(power_c4_over_time[i1:i2+1]) * (1/160)

        e_ratio_c3 = e_c3/e_baseline_c3
        e_ratio_c4 = e_c4/e_baseline_c4

        baseline_p_c3 = power_c3_over_time_baseline[:560].mean()
        baseline_p_c4 = power_c4_over_time_baseline[:560].mean()
        mi_p_c3 = power_c3_over_time[80:640].mean()
        mi_p_c4 = power_c4_over_time[80:640].mean()


        if case == "ll":
            ers_c3_ll.append(ers_c3_max)
            ers_c4_ll.append(ers_c4_max)
            e_c3_ll.append(e_ratio_c3)
            e_c4_ll.append(e_ratio_c4)
            baseline_p_c3_ll.append(baseline_p_c3)
            baseline_p_c4_ll.append(baseline_p_c4)
            mi_p_c3_ll.append(mi_p_c3)
            mi_p_c4_ll.append(mi_p_c4)
        else:
            ers_c3_rl.append(ers_c3_max)
            ers_c4_rl.append(ers_c4_max)
            e_c3_rl.append(e_ratio_c3)
            e_c4_rl.append(e_ratio_c4)
            baseline_p_c3_rl.append(baseline_p_c3)
            baseline_p_c4_rl.append(baseline_p_c4)
            mi_p_c3_rl.append(mi_p_c3)
            mi_p_c4_rl.append(mi_p_c4)
    return

def ers_restructuring():
    group_A = [i-1 for i in subs_pattern_A]
    group_B = [i-1 for i in subs_pattern_B]
    group_weak = [i-1 for i in range(1,18) if i not in subs_pattern_A and i not in subs_pattern_B]
    
    # pattern A
    ers_c3_ll_A = [ers_c3_ll[i] for i in group_A]
    ers_c4_rl_A = [ers_c4_rl[i] for i in group_A]

    # pattern B
    ers_c3_ll_B = [ers_c3_ll[i] for i in group_B]
    ers_c4_rl_B = [ers_c4_rl[i] for i in group_B]

    # pattern other
    ers_c3_ll_weak = [ers_c3_ll[i] for i in group_weak ]
    ers_c4_rl_weak = [ers_c4_rl[i] for i in group_weak ]

    pairs = [[ers_c3_ll_A, ers_c3_ll_B,ers_c3_ll_weak], [ers_c4_rl_A, ers_c4_rl_B,ers_c4_rl_weak]]

    return (pairs,ers_c3_ll_A, ers_c3_ll_B,ers_c3_ll_weak,ers_c4_rl_A, ers_c4_rl_B,ers_c4_rl_weak )
def e_restructuring():
    group_A = [i-1 for i in subs_pattern_A]
    group_B = [i-1 for i in subs_pattern_B]
    group_weak = [i-1 for i in range(1,18) if i not in subs_pattern_A and i not in subs_pattern_B]
    
    # pattern A
    e_c3_ll_A = [e_c3_ll[i] for i in group_A]
    e_c4_rl_A = [e_c4_rl[i] for i in group_A]

    # pattern B
    e_c3_ll_B = [e_c3_ll[i] for i in group_B]
    e_c4_rl_B = [e_c4_rl[i] for i in group_B]

    # pattern other
    e_c3_ll_weak = [e_c3_ll[i] for i in group_weak ]
    e_c4_rl_weak = [e_c4_rl[i] for i in group_weak ]

    pairs_e = [[e_c3_ll_A, e_c3_ll_B,e_c3_ll_weak], [e_c4_rl_A, e_c4_rl_B,e_c4_rl_weak]]

    return (pairs_e,e_c3_ll_A, e_c3_ll_B,e_c3_ll_weak,e_c4_rl_A, e_c4_rl_B,e_c4_rl_weak )
def ers_whisker_plotting(pairs):
    c="LL"
    for data in pairs:
        plt.figure(figsize=(4, 6))
        box = plt.boxplot(
            data,
            positions=[.5, 1, 1.5],
            widths=0.3,
            patch_artist=True,
            labels=['A', 'B', 'weak']
        )
        colors = ['lightyellow', 'lightblue', 'red']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        for median in box['medians']:
            median.set_color('black')
        plt.tick_params(axis='y', labelsize=14)

        plt.xticks([0.5, 1, 1.5], ['A', 'B', 'weak'], fontsize=12)
        plt.xlabel('')
        plt.tight_layout()
        plt.savefig(fr"{c}_ers_box.png")
        c="RL"
def export_ers(ers_c3_ll_A, ers_c3_ll_B,ers_c3_ll_weak,ers_c4_rl_A, ers_c4_rl_B,ers_c4_rl_weak ):
    pd.DataFrame({
        "ers_rl_A": pd.Series(ers_c4_rl_A),
        "ers_ll_A": pd.Series(ers_c3_ll_A),
        "ers_ll_B": pd.Series(ers_c3_ll_B),
        "ers_rl_B": pd.Series(ers_c4_rl_B),
        "ers_ll_weak": pd.Series(ers_c3_ll_weak),
        "ers_rl_weak": pd.Series(ers_c4_rl_weak),
    }).to_csv("ers_values_dataset1.csv", index=False)  

    return 
def ers_plotting():
    def _avg_cat(baseline_list, mi_list):
        b = np.mean(np.asarray(baseline_list), axis=0)
        m = np.mean(np.asarray(mi_list), axis=0)
        return np.concatenate([b, m], axis=0)

    groups = {
        "A": (power_c3_over_time_baseline_A, power_c3_over_time_A,
              power_c4_over_time_baseline_A, power_c4_over_time_A),
        "B": (power_c3_over_time_baseline_B, power_c3_over_time_B,
              power_c4_over_time_baseline_B, power_c4_over_time_B),
        "W": (power_c3_over_time_baseline_W, power_c3_over_time_W,
              power_c4_over_time_baseline_W, power_c4_over_time_W),
    }

    with open("groups.pkl", "wb") as f:
        pickle.dump(groups, f)

    for lbl, (c3_base, c3_mi, c4_base, c4_mi) in groups.items():
        data_ready_1 = _avg_cat(c3_base, c3_mi)  # LL
        data_ready_2 = _avg_cat(c4_base, c4_mi)  # RL

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

        for ax, data, title in zip(axes, 
                                   [data_ready_1, data_ready_2], 
                                   ["Left Limb (LL)", "Right Limb (RL)"]):
            ax.plot(data)
            ax.axvline(x=640, linestyle="--", color="k")              # vertical line
            ax.axvspan(160, 800, color="black", alpha=0.5)           # shaded box

            # Labels
            ymin, ymax = ax.get_ylim()
            ymid = ymin + 0.9*(ymax - ymin)   # place labels near top

            ax.text(125, ymid, "Baseline", ha="center", va="center",
                    fontsize=10, color="darkgreen", weight="bold")
            ax.text(500, ymid, "Transition:\nignored", ha="center", va="center",
                    fontsize=9, color="white", weight="bold")
            ax.text(1100, ymid, "MI", ha="center", va="center",
                    fontsize=10, color="darkred", weight="bold")

            ax.set_title(f"{lbl} - Averaged time series - {title}")
            ax.set_ylabel("Amplitude")

        axes[1].set_xlabel("Time")
        plt.tight_layout()
        plt.savefig(f"{lbl}_erss.svg", format="svg")
        plt.close(fig)

def baseline_p_restructuring():
    group_A = [i-1 for i in subs_pattern_A]
    group_B = [i-1 for i in subs_pattern_B]
    group_weak = [i-1 for i in range(1,18) if i not in subs_pattern_A and i not in subs_pattern_B]
    
    # pattern A
    baseline_p_c3_ll_A = [baseline_p_c3_ll[i] for i in group_A]
    baseline_p_c4_rl_A = [baseline_p_c4_rl[i] for i in group_A]

    # pattern B
    baseline_p_c3_ll_B = [baseline_p_c3_ll[i] for i in group_B]
    baseline_p_c4_rl_B = [baseline_p_c4_rl[i] for i in group_B]

    # pattern other
    baseline_p_c3_ll_weak = [baseline_p_c3_ll[i] for i in group_weak ]
    baseline_p_c4_rl_weak = [baseline_p_c4_rl[i] for i in group_weak ]

    pairs_x = [[baseline_p_c3_ll_A, baseline_p_c3_ll_B,baseline_p_c3_ll_weak], [baseline_p_c4_rl_A, baseline_p_c4_rl_B,baseline_p_c4_rl_weak]]

    return (pairs_x,baseline_p_c3_ll_A, baseline_p_c3_ll_B,baseline_p_c3_ll_weak, baseline_p_c4_rl_A, baseline_p_c4_rl_B,baseline_p_c4_rl_weak)
def mi_p_restructuring():
    group_A = [i-1 for i in subs_pattern_A]
    group_B = [i-1 for i in subs_pattern_B]
    group_weak = [i-1 for i in range(1,18) if i not in subs_pattern_A and i not in subs_pattern_B]
    
    # pattern A
    mi_p_c3_ll_A = [mi_p_c3_ll[i] for i in group_A]
    mi_p_c4_rl_A = [mi_p_c4_rl[i] for i in group_A]

    # pattern B
    mi_p_c3_ll_B = [mi_p_c3_ll[i] for i in group_B]
    mi_p_c4_rl_B = [mi_p_c4_rl[i] for i in group_B]

    # pattern other
    mi_p_c3_ll_weak = [mi_p_c3_ll[i] for i in group_weak ]
    mi_p_c4_rl_weak = [mi_p_c4_rl[i] for i in group_weak ]

    pairs_y = [[mi_p_c3_ll_A, mi_p_c3_ll_B,mi_p_c3_ll_weak], [mi_p_c4_rl_A, mi_p_c4_rl_B,mi_p_c4_rl_weak]]

    return (pairs_y,mi_p_c3_ll_A, mi_p_c3_ll_B,mi_p_c3_ll_weak, mi_p_c4_rl_A, mi_p_c4_rl_B,mi_p_c4_rl_weak)

def export_e(e_c3_ll_A, e_c3_ll_B,e_c3_ll_weak,e_c4_rl_A, e_c4_rl_B,e_c4_rl_weak ):

    pd.DataFrame({
        "e_rl_A": pd.Series(e_c4_rl_A),
        "e_ll_A": pd.Series(e_c3_ll_A),
        "e_ll_B": pd.Series(e_c3_ll_B),
        "e_rl_B": pd.Series(e_c4_rl_B),
        "e_ll_weak": pd.Series(e_c3_ll_weak),
        "e_rl_weak": pd.Series(e_c4_rl_weak),
    }).to_csv("e_values_dataset1.csv", index=False)  

    return 
def export_baseline_p(baseline_p_c3_ll_A, baseline_p_c3_ll_B,baseline_p_c3_ll_weak, baseline_p_c4_rl_A, baseline_p_c4_rl_B,baseline_p_c4_rl_weak):

    pd.DataFrame({
        "baseline_p_c4_rl_A": pd.Series(baseline_p_c4_rl_A),
        "baseline_p_c3_ll_A": pd.Series(baseline_p_c3_ll_A),
        "baseline_p_c3_ll_B": pd.Series(baseline_p_c3_ll_B),
        "baseline_p_c4_rl_B": pd.Series(baseline_p_c4_rl_B),
        "baseline_p_c3_ll_weak": pd.Series(baseline_p_c3_ll_weak),
        "baseline_p_c4_rl_weak": pd.Series(baseline_p_c4_rl_weak),
    }).to_csv("baseline_p_values_dataset1.csv", index=False)  

    return 

def export_mi_p(mi_p_c3_ll_A, mi_p_c3_ll_B,mi_p_c3_ll_weak, mi_p_c4_rl_A, mi_p_c4_rl_B,mi_p_c4_rl_weak):

    pd.DataFrame({
        "mi_p_c4_rl_A": pd.Series(mi_p_c4_rl_A),
        "mi_p_c3_ll_A": pd.Series(mi_p_c3_ll_A),
        "mi_p_c3_ll_B": pd.Series(mi_p_c3_ll_B),
        "mi_p_c4_rl_B": pd.Series(mi_p_c4_rl_B),
        "mi_p_c3_ll_weak": pd.Series(mi_p_c3_ll_weak),
        "mi_p_c4_rl_weak": pd.Series(mi_p_c4_rl_weak),
    }).to_csv("mi_p_values_dataset1.csv", index=False)  

    return 

#Load data
physionet_paths = [ mne.datasets.eegbci.load_data(id,IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,"/root/mne_data",) for id in range(1, needed_subs + 1)  ]
physionet_paths = np.concatenate(physionet_paths)
raws = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING',) for path in physionet_paths]

# Process runs in groups of 3 (each subject)
ers_c3_ll=[]
ers_c4_ll=[]
ers_c3_rl=[]
ers_c4_rl=[]
e_c3_ll=[]
e_c4_ll=[]
e_c3_rl=[]
e_c4_rl=[]
baseline_p_c3_ll=[]
baseline_p_c4_ll=[]
mi_p_c3_ll=[]
mi_p_c4_ll=[]
baseline_p_c3_rl=[]
baseline_p_c4_rl=[]
mi_p_c3_rl=[]
mi_p_c4_rl=[]
power_c3_over_time_A=[]
power_c4_over_time_A=[]
power_c3_over_time_baseline_A=[]
power_c4_over_time_baseline_A=[]
power_c3_over_time_B=[]
power_c4_over_time_B=[]
power_c3_over_time_baseline_B=[]
power_c4_over_time_baseline_B=[]
power_c3_over_time_W=[]
power_c4_over_time_W=[]
power_c3_over_time_baseline_W=[]
power_c4_over_time_baseline_W=[]
delta_x_dic={}
delta_y_dic={}
psd_left_all_subs = {}
psd_right_all_subs = {}
stft_left_all_subs = {}
stft_right_all_subs = {}
labels_all_subs = {}

# Create directory to save figures
OUT_DIR = Path("MI Plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for beg_global_run_idx in range(0, len(raws), num_runs_sub):
    sub_mod = beg_global_run_idx // num_runs_sub + 1
    labels_MI_l=[]
    x_left_hemi_l=[]
    x_right_hemi_l=[]   
    psd_left_l=[]
    psd_right_l=[] 
    epochs_l=[]
    epochs_baseline_l=[]
    baseline_epochs_l=[]
    labels_MI_ers=[]
    for local_run_index in range(num_runs_sub): 
        global_run_idx = beg_global_run_idx + local_run_index
        if global_run_idx >= len(raws):
            break  
        raw = raws[global_run_idx]

        epoched,epochs_MI_run,epoched_baseline,epochs_baseline_run=raw_MI_processing(raw)
        x_left_hemi,x_right_hemi,y,psd_left,psd_right=MI_data_generation(epoched)

        labels_MI_l.append(y)
        x_left_hemi_l.append(x_left_hemi) 
        x_right_hemi_l.append(x_right_hemi) 
        psd_left_l.append(psd_left)
        psd_right_l.append(psd_right)
        epochs_l.append(epochs_MI_run) 
        epochs_baseline_l.append(epochs_baseline_run)

    labels_all=np.concatenate(labels_MI_l, axis=0)
    MI_tf_left_hemi=np.concatenate(x_left_hemi_l, axis=0)
    MI_tf_right_hemi=np.concatenate(x_right_hemi_l, axis=0)
    psd_left_ses=np.concatenate(psd_left_l, axis=0)
    psd_right_ses=np.concatenate(psd_right_l, axis=0)
    epochses=np.concatenate(epochs_l, axis=0)
    epochses_baseline=np.concatenate(epochs_baseline_l, axis=0)

    ers_process(ll_lbl=0, rl_lbl=1)

    if MI_essence == "PSD":
        psd_left_all_subs[sub_mod]=psd_left_ses
        psd_right_all_subs[sub_mod]=psd_right_ses
        labels_all_subs[sub_mod]=labels_all
        if generate_MI_plots:plotting_MI_datapoints(psd_left_ses,psd_right_ses,labels_all) # PSD

    elif MI_essence == "STFT":    
        stft_left_all_subs[sub_mod]=MI_tf_left_hemi
        stft_right_all_subs[sub_mod]=MI_tf_right_hemi
        labels_all_subs[sub_mod]=labels_all
        if generate_MI_plots: plotting_MI_datapoints(MI_tf_left_hemi,MI_tf_right_hemi,labels_all)  # tf
    


if MI_essence == "PSD":
    data_angs_dist = find_angs_dist(psd_left_all_subs,psd_right_all_subs,labels_all_subs)
    data_ang_dist_restructured = restructuring(data_angs_dist)  
    plotting_MI_datapoints_all(data_ang_dist_restructured)  
    plot_MI_whiskers(data_angs_dist)
    export_mi_data_csv(data_angs_dist)

elif MI_essence == "STFT":  
    data_angs_dist = find_angs_dist(stft_left_all_subs,stft_right_all_subs,labels_all_subs)
    data_ang_dist_restructured = restructuring(data_angs_dist)  
    plotting_MI_datapoints_all(data_ang_dist_restructured)  
    plot_MI_whiskers(data_angs_dist)
    export_mi_data_csv(data_angs_dist)

ers_plotting()
pairs,ers_c3_ll_A, ers_c3_ll_B,ers_c3_ll_weak,ers_c4_rl_A, ers_c4_rl_B,ers_c4_rl_weak  = ers_restructuring() 
pairs_e,e_c3_ll_A, e_c3_ll_B,e_c3_ll_weak,e_c4_rl_A, e_c4_rl_B,e_c4_rl_weak  = e_restructuring() 
pairs_x,baseline_p_c3_ll_A, baseline_p_c3_ll_B,baseline_p_c3_ll_weak, baseline_p_c4_rl_A, baseline_p_c4_rl_B,baseline_p_c4_rl_weak=baseline_p_restructuring()
pairs_y,mi_p_c3_ll_A, mi_p_c3_ll_B,mi_p_c3_ll_weak, mi_p_c4_rl_A, mi_p_c4_rl_B,mi_p_c4_rl_weak=mi_p_restructuring()
ers_whisker_plotting(pairs)
export_ers(ers_c3_ll_A, ers_c3_ll_B,ers_c3_ll_weak,ers_c4_rl_A, ers_c4_rl_B,ers_c4_rl_weak )
export_e(e_c3_ll_A, e_c3_ll_B,e_c3_ll_weak,e_c4_rl_A, e_c4_rl_B,e_c4_rl_weak )
export_baseline_p(baseline_p_c3_ll_A, baseline_p_c3_ll_B,baseline_p_c3_ll_weak, baseline_p_c4_rl_A, baseline_p_c4_rl_B,baseline_p_c4_rl_weak)
export_mi_p(mi_p_c3_ll_A, mi_p_c3_ll_B,mi_p_c3_ll_weak, mi_p_c4_rl_A, mi_p_c4_rl_B,mi_p_c4_rl_weak)


stop=1