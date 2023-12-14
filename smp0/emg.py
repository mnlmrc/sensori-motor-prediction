import os

import numpy as np
import pandas as pd
import smp0.globals as gl
from matplotlib import pyplot as plt
from scipy.signal import resample

from smp0.utils import hp_filter


def load_delsys(experiment=None, participant_id=None, block=None, muscle_names=None, trigger_name=None):
    """returns a pandas DataFrame with the raw EMG data recorded using the Delsys system

    :param participant_id:
    :param experiment:
    :param block:
    :param muscle_names:
    :param trigger_name:
    :return:
    """
    fname = f"{experiment}_{participant_id}_{block}.csv"
    filepath = os.path.join(gl.make_dirs(experiment, participant_id, "emg"), fname)

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


def emg_hp_filter(data, n_ord=None, cutoff=None, fsample=None, muscle_names=None):
    """

    :param data:
    :param n_ord:
    :param cutoff:
    :param fsample:
    :param muscle_names:
    :return:
    """
    data_filtered = pd.DataFrame()
    for col in muscle_names:
        data_filtered[col] = hp_filter(data[col], n_ord=n_ord, cutoff=cutoff, fsample=fsample)

    return data_filtered


def emg_rectify(data, muscle_names=None):
    """

    :param data:
    :param muscle_names:
    :return:
    """
    data_rectified = pd.DataFrame()
    for col in muscle_names:
        data_rectified[col] = data[col].abs()  # Rectify

    return data_rectified


def detect_trig(trig_sig, time_trig, amp_threshold=None, ntrials=None, debugging=False):
    """

    :param trig_sig:
    :param time_trig:
    :param amp_threshold:
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

    trig_sig = pd.to_numeric(trig_sig).to_numpy()
    time_trig = pd.to_numeric(time_trig).to_numpy()

    trig_sig[trig_sig < amp_threshold] = 0
    trig_sig[trig_sig > amp_threshold] = 1

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
    muscle_names = data.columns[:-1]
    n_muscles = len(muscle_names)
    ntrials = len(timestamp)
    timepoints = int(fsample * (prestim + poststim))

    emg_segmented = np.zeros((ntrials, n_muscles, timepoints))
    for tr, idx in enumerate(timestamp):
        for m, muscle in enumerate(muscle_names):
            emg_segmented[tr, m] = data[muscle][idx - int(prestim * fsample):
                                                idx + int(poststim * fsample)].to_numpy()

    return emg_segmented
