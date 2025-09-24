import os
import sys
import numpy as np
import pyedflib
from glob import glob

sys.path.append('gumpy')
from gumpy import signal


def preprocess_data(data, sample_rate, bp_low, bp_high, notch_f, normalize):
    data = signal.notch(data, notch_f / (sample_rate / 2))
    data = signal.butter_bandpass(data, bp_low, bp_high, order=5, fs=sample_rate)
    if normalize:
        data = signal.normalize(data, 'mean_std')
    return data


def load_data(FNAMES, base_folder, sample_rate,
              samples, preprocessing, bp_low, bp_high, notch_f,
              normalize, num_trials_per_run):

    # Remove subjects with bad data
    for bad_subj in ['S038', 'S088', 'S089', 'S092', 'S100', 'S104']:
        FNAMES = [sub for sub in FNAMES if sub != bad_subj]

    convert_label = lambda x: 0 if x == 'T1' else 1 if x == 'T2' else None
    file_numbers = ['04', '08', '12']

    X, y = [], []

    for subj in FNAMES:
        fnames = [
            f for f in glob(os.path.join(base_folder, subj, subj + 'R*.edf'))
            if f[-6:-4] in file_numbers
        ]

        for file_name in fnames:
            reader = pyedflib.EdfReader(file_name)
            times, durations, tasks = reader.readAnnotations()
            sigbufs = np.array([reader.readSignal(i) for i in range(reader.signals_in_file)])

            trial_data = np.zeros((num_trials_per_run, 64, samples))
            labels, signal_start, k = [], 0, 0

            for time, duration, task in zip(times, durations, tasks):
                if k == num_trials_per_run:
                    break
                if task == 'T0':
                    signal_start += int(sample_rate * duration)
                    continue

                signal_end = signal_start + samples
                for j in range(sigbufs.shape[0]):
                    segment = sigbufs[j, signal_start:signal_end]
                    if preprocessing:
                        segment = preprocess_data(segment, sample_rate, bp_low, bp_high,
                                                  notch_f, normalize)
                    trial_data[k, j] = segment

                data_trial = np.squeeze(trial_data[k, :]).T
                trial_data[k, :, :] = data_trial.T

                label = convert_label(task)
                if label is None:
                    raise ValueError(f"Invalid label {task}")
                labels.append(label)

                signal_start += int(sample_rate * duration)
                k += 1

            y.extend(labels)
            X.extend(trial_data.swapaxes(1, 2))

    X = np.stack(X)
    y = np.array(y).reshape(-1, 1)

    print("Loaded data shapes:", X.shape, y.shape)
    keep = [1, 2, 8, 9, 15, 16, 4, 5, 11, 12, 18, 19] # channels needed

    return X[:, :, keep], y
