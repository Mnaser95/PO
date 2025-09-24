import numpy as np
import mne
import sys
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from collections import Counter
import pandas as pd
import statsmodels.api as sm
import random
import os
import tensorflow as tf



def loading(ses, data_type,run):
    my_file = fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\Data\2a2b data\full_2a_data\Data\{ses-1}.mat"
    mat_data = scipy.io.loadmat(my_file)
    if data_type == 'rest':
        my_data_eeg = np.squeeze(mat_data['data'][0][1][0][0][0][:, 0:22]) # the first 22 channels are EEG
        my_data_eog = np.squeeze(mat_data['data'][0][1][0][0][0][:, 22:25]) # the rest are EOG
    elif data_type == 'mi':
        my_data_eeg = np.squeeze(mat_data['data'][0][run+3][0][0][0][:, 0:22])
        my_data_eog = np.squeeze(mat_data['data'][0][run+3][0][0][0][:, 22:25])
    return np.hstack([my_data_eeg, my_data_eog]),mat_data
def create_mne_raw(data,sample_rate):
    numbers = list(range(1, 26))
    ch_names = [str(num) for num in numbers]
    ch_types = ['eeg'] * 22 + ['eog'] * 3
    info = mne.create_info(ch_names=ch_names, sfreq=sample_rate, ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info)
    return raw
def process_mi_data(raw, mat_data, bp_low, bp_high, notch_f, tmin_MI, tmax_MI,run,normalize):
    raw.filter(bp_low, bp_high, fir_design='firwin') # FIR filtration to keep a range of frequencies
    raw.notch_filter(notch_f, fir_design='firwin') # FIR filtration to keep a range of frequencies

    events = np.squeeze(mat_data['data'][0][run+3][0][0][2]) # only the first run of each session is taken (total number of trials is 48, only left and right hand considered so 24)
    event_indices = np.squeeze(mat_data['data'][0][run+3][0][0][1])
    mne_events = np.column_stack((event_indices, np.zeros_like(event_indices), events))

    event_id_MI = dict({'769': 1, '770': 2})
    epochs_MI = mne.Epochs(raw, mne_events, event_id_MI, tmin_MI, tmax_MI, proj=True,  baseline=None, preload=True)
    labels_MI = epochs_MI.events[:, -1]
    data_MI_original = epochs_MI.get_data()

    if normalize:
        for i in range(data_MI_original.shape[0]):  # iterate over trials
            for j in range(data_MI_original.shape[1]):  # iterate over channels
                mean = np.mean(data_MI_original[i, j, :])
                std = np.std(data_MI_original[i, j, :])
                if std == 0:
                    std = 1  # avoid division by zero
                data_MI_original[i, j, :] = (data_MI_original[i, j, :] - mean) / std

    return (labels_MI,data_MI_original)



def load_data(subs_considered, sample_rate, bp_low, bp_high, notch_f,normalize, tmin_MI, tmax_MI):

    all_ses_data_list=[]
    all_ses_labels_list=[]

    for ses in subs_considered:
        all_run_data=[]
        all_run_labels=[]
        for run in range(0,3):
            mi_data, mat_data = loading(ses, 'mi',run)
            raw = create_mne_raw(mi_data,sample_rate)

            labels_MI,data_MI_original = process_mi_data(raw, mat_data, bp_low, bp_high, notch_f, tmin_MI, tmax_MI,run,normalize)
            all_run_data.append(data_MI_original)     
            all_run_labels.append(labels_MI)  
        all_ses_data=np.concatenate(all_run_data,axis=0)
        all_ses_labels=np.concatenate(all_run_labels,axis=0)

        all_ses_data_list.append(all_ses_data)
        all_ses_labels_list.append(all_ses_labels)

    all_ses_data_arr=np.array(all_ses_data_list)    
    all_ses_labels_arr=np.array(all_ses_labels_list)   

    all_ses_data_arr_reshaped=all_ses_data_arr.reshape(-1,all_ses_data_arr.shape[2],all_ses_data_arr.shape[3])
    data_ready=all_ses_data_arr_reshaped.swapaxes( 1, 2)
    labels_ready=all_ses_labels_arr.reshape(-1)

    keep = [1, 2, 7, 8, 13, 14, 4, 5, 10, 11, 16, 17] # channels needed

    return (data_ready[:, :, keep],labels_ready)

