import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import resample, firwin, filtfilt

from smp0.utils import hp_filter


def load_delsys(filepath, muscle_names=None, trigger_name=None):
    """returns a pandas DataFrame with the raw EMG data recorded using the Delsys system

    :param filepath: path to the .csv data exported from Delsys Trigno software
    :param muscle_names: 
    :param trigger_name:
    :return:
    """
    # read data from .csv file (Delsys output)
    with open(filepath, 'rt') as fid:
        A = []
        for line in fid:
            # Strip whitespace and newline characters, then split
            split_line = [elem.strip() for elem in line.strip().split(',')]
            A.append(split_line)

    # identify columns with data from each muscle
    muscle_columns = {}
    for muscle in muscle_names:
        for c, col in enumerate(A[3]):
            if muscle in col:
                muscle_columns[muscle] = c + 1  # EMG is on the right of Timeseries data (that's why + 1)
                break
        for c, col in enumerate(A[5]):
            if muscle in col:
                muscle_columns[muscle] = c + 1
                break

    df_raw = pd.DataFrame(A[7:])  # get rid of header
    df_out = pd.DataFrame()  # init final dataframe

    for muscle in muscle_columns:
        df_out[muscle] = pd.to_numeric(df_raw[muscle_columns[muscle]],
                                       errors='coerce').replace('', np.nan).dropna()  # add EMG to dataframe

    # add trigger column
    trigger_column = None
    for c, col in enumerate(A[3]):
        if trigger_name in col:
            trigger_column = c + 1

    try:
        trigger = df_raw[trigger_column]
        trigger = resample(trigger.values, len(df_out))
    except IOError as e:
        raise IOError("Trigger not found") from e

    df_out[trigger_name] = trigger

    # add time column
    df_out['time'] = df_raw.loc[:, 0]

    return df_out


def emg_hp_filter(data, muscle_names=None):
    """

    :param data:
    :param muscle_names:
    :return:
    """
    data_filtered = {}
    for col in muscle_names:
        data[col] = data[col]  # convert to floats
        data_filtered[col] = hp_filter(data[col])

    return data_filtered


def emg_rectify(data, muscle_names=None):
    """

    :param data:
    :param muscle_names:
    :return:
    """
    data_rectified = {}
    for col in muscle_names:
        data[col] = hp_filter(data[col])
        data_rectified[col] = data[col].abs()  # Rectify

    return data_rectified


def detect_trig(trig_sig, time_trig, ntrials=None, debugging=False):
    """
    Detects rising edge triggers for segmentation

    :param trig_sig:
    :param time_trig:
    :param ntrials:
    :param debugging:
    :return:
    """

    ########## old trigger detection (subj 100-101)
    # trig_sig = trig_sig / np.max(trig_sig)
    # diff_trig = np.diff(trig_sig)
    # diff_trig[diff_trig < self.amp_threshold] = 0
    # locs, _ = find_peaks(diff_trig)
    ##############################################

    trig_sig[trig_sig < self.amp_threshold] = 0
    trig_sig[trig_sig > self.amp_threshold] = 1

    # Detecting the edges
    diff_trig = np.diff(trig_sig)

    locs = np.where(diff_trig == 1)[0]

    # Debugging plots
    if debugging:
        # Printing the number of triggers detected and number of trials
        print("\nNum Trigs Detected = {}".format(len(locs)))
        print("Num Trials in Run = {}".format(ntrials))
        print("====NumTrial should be equal to NumTrigs====\n\n\n")

        # plotting block
        plt.figure()
        plt.plot(trig_sig, 'k', linewidth=1.5)
        plt.plot(diff_trig, '--r', linewidth=1)
        plt.scatter(locs, diff_trig[locs], color='red', marker='o', s=30)
        plt.xlabel("Time (index)")
        plt.ylabel("Trigger Signal (black), Diff Trigger (red dashed), Detected triggers (red/blue points)")
        plt.ylim([-1.5, 1.5])
        plt.show()

    # Getting rise and fall times and indexes
    rise_idx = locs
    rise_times = time_trig[rise_idx]

    # Sanity check
    if len(rise_idx) != ntrials:  # | (len(fall_idx) != Emg.ntrials):
        raise ValueError(f"Wrong number of trials: {len(rise_idx)}")

    return rise_times, rise_idx


def emg_segment(data, timestamp, prestim=None, poststim=None, fsample=None):
    """

    :param data:
    :param timestamp:
    :param prestim:
    :param poststim:
    :param fsample:
    :return:
    """
    emg_segmented = np.zeros((len(timestamp), len(self.muscle_names),
                              int(fsample * (self.prestim + self.poststim))))
    for tr, idx in enumerate(timestamp):
        for m, muscle in enumerate(self.muscle_names):
            emg_segmented[tr, m] = data[muscle][idx - int(prestim * fsample):
                                                idx + int(poststim * fsample)].to_numpy()

    return emg_segmented





