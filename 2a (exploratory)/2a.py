####################################### Libraries
import numpy as np
import mne
import scipy.io
from sklearn.metrics import accuracy_score
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from collections import Counter
from matplotlib.lines import Line2D
import pandas as pd
from scipy.signal import welch
from scipy.stats import kruskal
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from pathlib import Path
import csv
import pickle
##################################### Inputs
sfreq = 250 # sampling frequency
sess=[i for i in range(1,19)] # list of sessions (for all 9 subjects). Two sessions per subject so the total is 18.
picks = ["8", "9", "14", "15","2","3",   "11", "12", "17", "18", "5", "6"]     # the channels to consider (refer to data description)
f_low_rest=1   # low frequency
f_high_rest=20 # high frequency
f_low_MI=8   # low frequency
f_high_MI=12 # high frequency
tmin_rest = 1  # start of time for rest[s]
tmax_rest = 59 # end time for rest [s]
tmin_MI = 1
tmax_MI = 4
needed_ratio=1.4
generate_rest_maps = True
generate_MI_plots = True
MI_essence = "PSD" # either "PSD" or "STFT"

############################################################################## Clustering
def load_data(ses, data_type):
    my_file = fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\Data\2a2b data\full_2a_data\Data\{ses-1}.mat"
    mat_data = scipy.io.loadmat(my_file)
    if data_type == 'rest': # it's actually 2min, I am taking the first min
        my_data_eeg = np.squeeze(mat_data['data'][0][1][0][0][0][:, 0:22]) # the first 22 channels are EEG
        my_data_eog = np.squeeze(mat_data['data'][0][1][0][0][0][:, 22:25]) # the rest are EOG
    elif data_type == 'mi':
        my_data_eeg = np.squeeze(mat_data['data'][0][run+3][0][0][0][:, 0:22])
        my_data_eog = np.squeeze(mat_data['data'][0][run+3][0][0][0][:, 22:25])
    return np.hstack([my_data_eeg, my_data_eog]),mat_data
def create_mne_raw(data):
    numbers = list(range(1, 26))
    ch_names = [str(num) for num in numbers]
    ch_types = ['eeg'] * 22 + ['eog'] * 3
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info)
    return raw
def source_computing(raw):
    df = pd.read_csv(fr"25montage.csv", header=None, names=["name", "x", "y", "z"])
    ch_pos = {
        str(row['name']): np.array([row['x'], row['y'], row['z']])
        for _, row in df.iterrows()
    }

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing="warn")

    valid_chs = [
        ch['ch_name'] for ch in raw.info['chs']
        if ch['loc'] is not None
        and not np.allclose(ch['loc'][:3], 0)
        and not np.isnan(ch['loc'][:3]).any()
]

    raw = raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)
    #raw.plot_sensors(show_names=True)  # just to verify positions
    return raw
def process_rest_data(raw):
    raw.filter(f_low_rest, f_high_rest, fir_design='firwin') # FIR filtration to keep a range of frequencies

    raw=source_computing(raw)

    epoch_length_samples = int((tmax_rest-tmin_rest) * raw.info['sfreq'])
    n_samples = len(raw)

    # Creating a one event for the Rest period
    event_times = np.arange(0, n_samples - epoch_length_samples, epoch_length_samples)
    events = np.column_stack((event_times, np.zeros_like(event_times, dtype=int), np.ones_like(event_times, dtype=int)))
    epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin_rest, tmax=tmax_rest, baseline=None, preload=True, picks=picks)

    half_ch=len(picks)//2
    data_rest_left  = np.mean(epochs.get_data()[0][:half_ch, :], axis=0) # the first 4 channels are in the left hemisphere
    data_rest_right  = np.mean(epochs.get_data()[0][half_ch:, :], axis=0) # the last 4 channels are in the right hemisphere
    
    _, _, Zxx_left = stft(data_rest_left, sfreq, nperseg=sfreq) # generating time-frequency map using STFT
    _, _, Zxx_right = stft(data_rest_right, sfreq, nperseg=sfreq) # generating time-frequency map using STFT
    
    Zxx_mag_diff=np.abs(Zxx_left)-np.abs(Zxx_right)
    p_mag_diff=(np.abs(Zxx_left)**2)-(np.abs(Zxx_right)**2)

    return Zxx_mag_diff,p_mag_diff
def subject_select(mid,other):
    segment_size = 10
    n_segments = len(mid) // segment_size
    #ratio = [mid[i]/other[i] for i in range(n_segments)]
    ratios = []
    votes=[]
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        avg_mid_abs = np.abs(np.mean(mid[start:end]))
        avg_other_abs = np.abs(np.mean(other[start:end]))
        ratio = (avg_mid_abs / avg_other_abs)
        if ratio > needed_ratio:
            if np.mean(mid[start:end])>0:
                votes.append("11") 
            else:
                votes.append("10")
        else:
            votes.append("0X")

    #"11": strong, positive (mainly yellow)
    #"10": s0trong, negative (mainly blue)
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
def plotting_rest_maps(data):
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
        vmin=-0.004,
        vmax=0.004
    )

    # Custom colorbar
    cbar = plt.colorbar(im)
    cbar.set_ticks([-0.004, 0, 0.004])
    cbar.set_ticklabels(["Min (-ve)", "0", "Max (+ve)"])
    cbar.ax.tick_params(labelsize=14)

    # Custom x-axis ticks (e.g., 5 ticks between 0 and 60)
    tick_labels = [0, 15, 30, 45, 60]
    tick_positions = [int(i * (data.shape[1] / 60)) for i in tick_labels]
    plt.xticks(tick_positions, [str(i) for i in tick_labels], fontsize=14)
    plt.yticks(fontsize=14)

    # Axis labels
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Frequency [Hz]", fontsize=18)

    # Optional: Add a title
    plt.title(fr"Rest: |STFT| left hemi - |STFT| right hemi", fontsize=18)

    plt.ylim(0, 20)
    plt.tight_layout()
    out_path =Path(OUT_DIR) / f"Rest_{ses:03d}.svg"
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close()

def plotting_rest_maps_all_subs(data_dict):
    def avg_group(ids):
        mats, used, missing = [], [], []
        for sid in ids:
            arr = data_dict.get(sid)
            if arr is None:
                missing.append(sid)
            else:
                mats.append(np.asarray(arr))
                used.append(sid)
        if not mats:
            return None, used, missing
        shapes = {m.shape for m in mats}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent shapes: {shapes}")
        return np.nanmean(np.stack(mats, axis=0), axis=0), used, missing

    def plot_and_save(avg_map, group_name, n_used):
        colors = [(0, 'blue'), (0.3, 'skyblue'), (0.5, 'black'),
                  (0.7, 'lightyellow'), (1, 'yellow')]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(avg_map, aspect='auto', cmap=cmap,
                        interpolation='nearest', vmin=-0.004, vmax=0.004)
        plt.colorbar(im)

        tick_labels = [0, 15, 30, 45, 60]
        tick_pos = [int(i * (avg_map.shape[1] / 60)) for i in tick_labels]
        plt.xticks(tick_pos, [str(i) for i in tick_labels], fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel("Time [s]", fontsize=18)
        plt.ylabel("Frequency [Hz]", fontsize=18)
        plt.title(f"Rest: |STFT| LH - RH • {group_name} • n={n_used}", fontsize=16)

        plt.ylim(0, 20)
        plt.tight_layout()

        path = f"Rest_avg_{group_name}.svg"
        plt.savefig(path)
        plt.close()
        return path

    groups = {
        "Pattern_A": list(subs_pattern_A),
        "Pattern_B": list(subs_pattern_B),
        "Weak":      list(subs_weak),
    }

    results = {}
    for gname, ids in groups.items():
        avg_map, used, missing = avg_group(ids)
        if avg_map is None:
            results[gname] = {"used_ids": [], "missing_ids": missing,
                              "path": None, "avg": None}
            continue
        path = plot_and_save(avg_map, gname, len(used))
        results[gname] = {"used_ids": used, "missing_ids": missing,
                          "path": path, "avg": avg_map}

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

# Create directory to save figures
OUT_DIR = Path("Rest Maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

patterns={}
data_rest_diff_tf_mag_all_subs = {}
for ses in sess:
    rest_data, _= load_data(ses, 'rest')
    raw = create_mne_raw(rest_data)
    data_rest_diff_tf_mag, data_rest_diff_p_mag = process_rest_data(raw) 
    
    avg_mid_freq=np.mean(data_rest_diff_tf_mag[7:13,:],axis=0)
    avg_below_freq=np.mean(data_rest_diff_tf_mag[2:7,:],axis=0)  

    res_down=subject_select(avg_mid_freq,avg_below_freq)

    data_rest_diff_tf_mag_all_subs[ses]=data_rest_diff_tf_mag
    if generate_rest_maps: plotting_rest_maps(data_rest_diff_tf_mag)

    patterns[ses] = res_down

# for k in [k for k, v in patterns.items() if v == "Weak"]: #remove weak
#     del patterns[k]
#     del confidences[k]

subs_taken=list(patterns.keys())
subs_pattern_B = [k for k, v in patterns.items() if v == "Pattern B"]
subs_pattern_A = [k for k, v in patterns.items() if v == "Pattern A"]
subs_weak = [k for k, v in patterns.items() if v == "Weak"]

plotting_rest_maps_all_subs(data_rest_diff_tf_mag_all_subs)
export_rest_data(data_rest_diff_tf_mag_all_subs)

############################################################################## Next
def align_epochs_by_selection(e1, e2):
        # selections are unique and refer to original events indices
        common_sel = np.intersect1d(e1.selection, e2.selection)  # sorted

        # maps from original-event index -> current position in each Epochs
        map1 = {sel: i for i, sel in enumerate(e1.selection)}
        map2 = {sel: i for i, sel in enumerate(e2.selection)}

        idx1 = [map1[s] for s in common_sel]
        idx2 = [map2[s] for s in common_sel]

        return e1[idx1], e2[idx2]
def process_mi_data(raw, mat_data):
    raw.filter(f_low_MI, f_high_MI, fir_design='firwin') # FIR filtration to keep a range of frequencies
    df = pd.read_csv(fr"25montage.csv", header=None, names=["name", "x", "y", "z"])
    ch_pos = {str(row['name']): np.array([row['x'], row['y'], row['z']])for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing="warn")
    valid_chs = [ch['ch_name'] for ch in raw.info['chs'] if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()]

    raw = raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)

    events = np.squeeze(mat_data['data'][0][run+3][0][0][2]) # only the first run of each session is taken (total number of trials is 48, only left and right hand considered so 24)
    event_indices = np.squeeze(mat_data['data'][0][run+3][0][0][1])
    mne_events = np.column_stack((event_indices, np.zeros_like(event_indices), events))

    event_id_MI = dict({'769': 1, '770': 2})
    full_epochs=mne.Epochs(raw, mne_events, event_id_MI, 0, 3, proj=True,  baseline=None, preload=True, picks=picks)
    base_epochs=mne.Epochs(raw, mne_events, event_id_MI, -3, 0, proj=True,  baseline=None, preload=True, picks=picks)
    
    full_epochs, base_epochs = align_epochs_by_selection(full_epochs, base_epochs) # this is needed as some trials don't have the full baseline period. Those are dropped here.

    labels_MI = full_epochs.events[:, -1]
    data_MI_original = full_epochs.get_data()

    _, _, Zxx = stft(data_MI_original, sfreq, nperseg=sfreq) # generating time-frequency map using STFT
    
    MI_tf=np.abs(Zxx)
    _, psd_run = welch(data_MI_original, fs=sfreq)

    return (labels_MI,MI_tf,psd_run,full_epochs,base_epochs)
def plotting_MI_datapoints(data_received, labels_MI):
    unique_labels = np.unique(labels_MI)

    # Prepare scatter plot
    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        label_text = "LL-MI" if label == 1 else "RL-MI"
        color = plt.cm.viridis((label - 1) / (len(unique_labels) - 1))

        # Plot individual points
        plt.scatter(
            data_received[labels_MI == label, 0],
            data_received[labels_MI == label, 1],
            c=[color],
            label=label_text,
            s=100
        )

        # Plot average point as a large 'X'
        avg_x = np.mean(data_received[labels_MI == label, 0])
        avg_y = np.mean(data_received[labels_MI == label, 1])
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
    plt.xlabel(r"$\mathrm{PSD}_{\mathrm{MI,\ LH}}$", fontsize=20)
    plt.ylabel(r"$\mathrm{PSD}_{\mathrm{MI,\ RH}}$", fontsize=20)
    plt.title("MI Distribution", fontsize=18)

    # Legend and grid
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    out_path = OUT_DIR / f"MI_{ses}.svg"
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
            for lbl in (1, 2):  # 1=LH, 2=RH
                m = (lb == lbl)
                if np.any(m):
                    pts[lbl] = (x[m].mean(), y[m].mean())

            if 1 in pts and 2 in pts:
                (x0, y0), (x1, y1) = pts[1], pts[2]
                dx, dy = x1 - x0, y1 - y0
                theta = np.degrees(np.arctan2(dy, dx));  theta = theta if theta >= 0 else theta + 360
                dist = float(np.hypot(dx, dy))

                out[gkey]["pairs"].append((x0, y0, x1, y1))
                out[gkey]["angles"].append(theta)
                out[gkey]["distances"].append(dist)
                out[gkey]["rows"].append({
                    "group": gkey, "subject": sid,
                    "x_LH": x0, "y_LH": y0, "x_RH": x1, "y_RH": y1,
                    "angle_deg": theta, "distance": dist, "dataset": "set2"
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
    if label_names is None:
        label_names = {1: "LH-MI", 2: "RH-MI"}
    if colors is None:
        colors = {1: plt.cm.viridis(0.0), 2: plt.cm.viridis(1.0)}

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
            plt.scatter(x0, y0, c=[colors[1]], s=200, marker='*',
                        edgecolors='black', linewidths=1.0,
                        label=label_names[1] if 1 not in plotted_lbl else None)
            plt.scatter(x1, y1, c=[colors[2]], s=200, marker='*',
                        edgecolors='black', linewidths=1.0,
                        label=label_names[2] if 2 not in plotted_lbl else None)
            plotted_lbl.update({1, 2})
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

def ers_process(ll_lbl, rl_lbl):
    for case, lbl_id in (("ll", ll_lbl), ("rl", rl_lbl)):

        ############################################## Calculating power values
        mi_mask = labels_MI == lbl_id
        mi_full = full_epochs[mi_mask]      # (n_epochs_MI, n_channels, n_timepoints)
        mi = np.stack(mi_full, axis=0)
        power_c3_over_epochs_and_time = mi[:, :6, :].mean(axis=1)**2   # (n_epochs, n_time)
        power_c4_over_epochs_and_time = mi[:, 6:12, :].mean(axis=1)**2
        power_c3_over_time = power_c3_over_epochs_and_time.mean(axis=0)  # (n_time,)
        power_c4_over_time = power_c4_over_epochs_and_time.mean(axis=0)

        baseline_full = full_base_all_epochs[mi_mask]
        mi_base = np.stack(baseline_full, axis=0)
        power_c3_over_epochs_and_time_baseline = mi_base[:, :6, :].mean(axis=1)**2   # (n_epochs, n_time)
        power_c4_over_epochs_and_time_baseline = mi_base[:, 6:12, :].mean(axis=1)**2
        power_c3_over_time_baseline = power_c3_over_epochs_and_time_baseline.mean(axis=0)  # (n_time,)
        power_c4_over_time_baseline = power_c4_over_epochs_and_time_baseline.mean(axis=0)

        ############################################## Saving power values
        if ses in subs_pattern_A:
            if case == "ll":
                power_c3_over_time_A.append(power_c3_over_time)
                power_c3_over_time_baseline_A.append(power_c3_over_time_baseline)
            else:
                power_c4_over_time_A.append(power_c4_over_time)
                power_c4_over_time_baseline_A.append(power_c4_over_time_baseline)
        elif ses in subs_pattern_B:
            if case == "ll":
                power_c3_over_time_B.append(power_c3_over_time)
                power_c3_over_time_baseline_B.append(power_c3_over_time_baseline)
            else:
                power_c4_over_time_B.append(power_c4_over_time)
                power_c4_over_time_baseline_B.append(power_c4_over_time_baseline)
        else:
            if case == "ll":
                power_c3_over_time_W.append(power_c3_over_time)
                power_c3_over_time_baseline_W.append(power_c3_over_time_baseline)
            else:
                power_c4_over_time_W.append(power_c4_over_time)
                power_c4_over_time_baseline_W.append(power_c4_over_time_baseline)

        ############################################## Calculating ERS values
        step = 250
        i_1 = 125
        i_2 = i_1 + step // 2
        ers_c3_max = -10000
        ers_c4_max = -10000
        while i_2 <= 750:
            a = ((power_c3_over_time[i_1:i_2].mean() - power_c3_over_time_baseline[:625].mean())
                 / power_c3_over_time_baseline[:625].mean()) * 100
            b = ((power_c4_over_time[i_1:i_2].mean() - power_c4_over_time_baseline[:625].mean())
                 / power_c4_over_time_baseline[:625].mean()) * 100
            print(a)
            ers_c3_max = max(a, ers_c3_max)
            ers_c4_max = max(b, ers_c4_max)

            i_1 += step // 2
            i_2 += step // 2

        i1, i2 = 0, 625
        e_baseline_c3 = np.sum(power_c3_over_time_baseline[i1:i2+1]) * (1/250)
        e_baseline_c4 = np.sum(power_c4_over_time_baseline[i1:i2+1]) * (1/250)
        i1, i2 = 125, 750
        e_c3 = np.sum(power_c3_over_time[i1:i2+1]) * (1/250)
        e_c4 = np.sum(power_c4_over_time[i1:i2+1]) * (1/250)

        e_ratio_c3 = e_c3/e_baseline_c3
        e_ratio_c4 = e_c4/e_baseline_c4

        baseline_p_c3 = power_c3_over_time_baseline[:625].mean()
        baseline_p_c4 = power_c4_over_time_baseline[:625].mean()
        mi_p_c3 = power_c3_over_time[125:750].mean()
        mi_p_c4 = power_c4_over_time[125:750].mean()







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

        for ax, data, limb in zip(axes, 
                                  [data_ready_1, data_ready_2], 
                                  ["Left Limb (LL)", "Right Limb (RL)"]):
            ax.plot(data)
            ax.axvline(x=750, linestyle="--", color="k")              # vertical line
            ax.axvspan(250, 1000, color="black", alpha=0.5)           # shaded box

            # Labels
            ymin, ymax = ax.get_ylim()
            ymid = ymin + 0.9*(ymax - ymin)   # place labels near top

            ax.text(125, ymid, "Baseline", ha="center", va="center",
                    fontsize=10, color="darkgreen", weight="bold")
            ax.text(625, ymid, "Transition:\nignored", ha="center", va="center",
                    fontsize=9, color="white", weight="bold")
            ax.text(1100, ymid, "MI", ha="center", va="center",
                    fontsize=10, color="darkred", weight="bold")

            # Title now includes pattern label
            ax.set_title(f"{lbl} - Averaged time series - {limb}")
            ax.set_ylabel("Amplitude")

        axes[1].set_xlabel("Time")
        plt.tight_layout()
        plt.savefig(f"{lbl}_erss.svg", format="svg")
        plt.close(fig)
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

def export_ers(ers_c3_ll_A, ers_c3_ll_B,ers_c3_ll_weak,ers_c4_rl_A, ers_c4_rl_B,ers_c4_rl_weak ):

    pd.DataFrame({
        "ers_rl_A": pd.Series(ers_c4_rl_A),
        "ers_ll_A": pd.Series(ers_c3_ll_A),
        "ers_ll_B": pd.Series(ers_c3_ll_B),
        "ers_rl_B": pd.Series(ers_c4_rl_B),
        "ers_ll_weak": pd.Series(ers_c3_ll_weak),
        "ers_rl_weak": pd.Series(ers_c4_rl_weak),
    }).to_csv("ers_values_dataset2.csv", index=False)  

    return 
def export_e(e_c3_ll_A, e_c3_ll_B,e_c3_ll_weak,e_c4_rl_A, e_c4_rl_B,e_c4_rl_weak ):

    pd.DataFrame({
        "e_rl_A": pd.Series(e_c4_rl_A),
        "e_ll_A": pd.Series(e_c3_ll_A),
        "e_ll_B": pd.Series(e_c3_ll_B),
        "e_rl_B": pd.Series(e_c4_rl_B),
        "e_ll_weak": pd.Series(e_c3_ll_weak),
        "e_rl_weak": pd.Series(e_c4_rl_weak),
    }).to_csv("e_values_dataset2.csv", index=False)  

    return 

def export_baseline_p(baseline_p_c3_ll_A, baseline_p_c3_ll_B,baseline_p_c3_ll_weak, baseline_p_c4_rl_A, baseline_p_c4_rl_B,baseline_p_c4_rl_weak):

    pd.DataFrame({
        "baseline_p_c4_rl_A": pd.Series(baseline_p_c4_rl_A),
        "baseline_p_c3_ll_A": pd.Series(baseline_p_c3_ll_A),
        "baseline_p_c3_ll_B": pd.Series(baseline_p_c3_ll_B),
        "baseline_p_c4_rl_B": pd.Series(baseline_p_c4_rl_B),
        "baseline_p_c3_ll_weak": pd.Series(baseline_p_c3_ll_weak),
        "baseline_p_c4_rl_weak": pd.Series(baseline_p_c4_rl_weak),
    }).to_csv("baseline_p_values_dataset2.csv", index=False)  

    return 

def export_mi_p(mi_p_c3_ll_A, mi_p_c3_ll_B,mi_p_c3_ll_weak, mi_p_c4_rl_A, mi_p_c4_rl_B,mi_p_c4_rl_weak):

    pd.DataFrame({
        "mi_p_c4_rl_A": pd.Series(mi_p_c4_rl_A),
        "mi_p_c3_ll_A": pd.Series(mi_p_c3_ll_A),
        "mi_p_c3_ll_B": pd.Series(mi_p_c3_ll_B),
        "mi_p_c4_rl_B": pd.Series(mi_p_c4_rl_B),
        "mi_p_c3_ll_weak": pd.Series(mi_p_c3_ll_weak),
        "mi_p_c4_rl_weak": pd.Series(mi_p_c4_rl_weak),
    }).to_csv("mi_p_values_dataset2.csv", index=False)  

    return 

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
psd_left_all_subs={}
psd_right_all_subs={}
stft_left_all_subs={}
stft_right_all_subs={}
labels_all_subs={}
labels_stat = []

# Create directory to save figures
OUT_DIR = Path("MI Plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for ses in sess:
    data_MI_l=[]
    labels_MI_l=[]
    MI_tf_l=[]
    psd_l=[]
    epochs_l=[]
    baseline_epochs_l=[]
    full_epochs_run_l=[]
    base_all_epochs=[]
    for run in range(0,3):
        
        mi_data, mat_data = load_data(ses, 'mi')
        raw = create_mne_raw(mi_data)
        labels_MI_run,MI_tf_run,psd_run,full_epochs_run,base_epochs = process_mi_data(raw, mat_data)

        labels_MI_l.append(labels_MI_run)
        MI_tf_l.append(MI_tf_run) 
        psd_l.append(psd_run) 
        full_epochs_run_l.append(full_epochs_run)
        base_all_epochs.append(base_epochs)

    labels_MI=np.concatenate(labels_MI_l, axis=0)
    MI_tf=np.concatenate(MI_tf_l, axis=0)
    psds=np.concatenate(psd_l, axis=0)
    full_epochs=np.concatenate(full_epochs_run_l, axis=0)
    full_base_all_epochs=np.concatenate(base_all_epochs, axis=0)

    ers_process(ll_lbl=1, rl_lbl=2)

    if MI_essence == "PSD":
        psds_avg_freq=np.mean(np.squeeze(psds),axis=2) 
        psds_avg_freq_left_hemi = np.mean(psds_avg_freq[:, :6], axis=1) 
        psds_avg_freq_right_hemi = np.mean(psds_avg_freq[:, 6:], axis=1) 
        psd_left_all_subs[ses]=psds_avg_freq_left_hemi
        psd_right_all_subs[ses]=psds_avg_freq_right_hemi
        labels_all_subs[ses]=labels_MI
        psds_stacked =  np.vstack((psds_avg_freq_left_hemi, psds_avg_freq_right_hemi)).T 
        if generate_MI_plots: plotting_MI_datapoints(psds_stacked,labels_MI) 
    elif MI_essence == "STFT":
        data_MI_tf_abs_avg_freq=np.mean(np.squeeze(MI_tf),axis=2) 
        data_MI_tf_abs_avg_freq_time=np.mean(np.squeeze(data_MI_tf_abs_avg_freq),axis=2) 
        data_MI_tf_abs_avg_freq_time_left_hemi=np.mean(data_MI_tf_abs_avg_freq_time[:, :6], axis=1) 
        data_MI_tf_abs_avg_freq_time_right_hemi=np.mean(data_MI_tf_abs_avg_freq_time[:, 6:], axis=1) 
        stft_left_all_subs[ses]=data_MI_tf_abs_avg_freq_time_left_hemi
        stft_right_all_subs[ses]=data_MI_tf_abs_avg_freq_time_right_hemi
        labels_all_subs[ses]=labels_MI
        data_mi_stacked_tf = np.vstack((data_MI_tf_abs_avg_freq_time_left_hemi, data_MI_tf_abs_avg_freq_time_right_hemi)).T  # Combine along the second axis
        if generate_MI_plots: plotting_MI_datapoints(data_mi_stacked_tf,labels_MI)

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