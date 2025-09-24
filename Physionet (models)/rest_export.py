def read_rest(sub):

    import mne
    from scipy.signal import stft
    import os
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
    import sys
    sys.path.append('gumpy')
    from gumpy import signal

    baseline_eye_closed = [2]
    needed_channels = [1, 2, 8, 9, 15, 16, 4, 5, 11, 12, 18, 19] # channels needed
    fs=160


    low_rest=1
    high_rest=20

    #################################################################################################################################################### Rest
        
    def downsample_linear(X, new_n):
        old_n = X.shape[0]
        t_old = np.linspace(0, 1, old_n, endpoint=False)
        t_new = np.linspace(0, 1, new_n, endpoint=False)
        return np.column_stack([np.interp(t_new, t_old, X[:, c]) for c in range(X.shape[1])])


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

        # raw = mne.preprocessing.compute_current_source_density(raw)
        # raw.plot_sensors(show_names=True)  # just to verify positions

        epoched = mne.Epochs(raw,events,event_id=dict(rest=1),tmin=0,tmax=60,proj=False,picks=needed_channels,baseline=None,preload=True)

        return epoched

    # Download/load data paths
    physionet_paths = [mne.datasets.eegbci.load_data(subject_id,baseline_eye_closed,"/root/mne_data" ) for subject_id in range(sub, sub + 1) ]
    physionet_paths = np.concatenate(physionet_paths)

    # Read EDF files
    parts = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING')for path in physionet_paths] # 0-indexed

    for sub, raw in enumerate(parts):
        epoched=raw_rest_processing(raw)
        rest_data=epoched.get_data()
        rest_data=rest_data.squeeze().T[1:,:] # shape: (9600,12)

        # rest_data_ds=rest_data.reshape(-1,rest_data.shape[0]/4,rest_data.shape[1]).mean(axis=1)
        # split the 60s into segmenets of 4s and find the average

        # rest_data_ds = rest_data[:640,:]
        # take the first 4s window

        rest_data_ds = downsample_linear(rest_data, 640)
        # basic downsampling

        # rest_data_ds = rest_data
        # no downsampling

        rest_data_ds = signal.normalize(rest_data_ds, 'mean_std')
        # normalization

    return(rest_data_ds)