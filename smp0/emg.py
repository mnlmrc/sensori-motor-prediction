import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

import matplotlib

matplotlib.use('MacOSX')


def detect_trig(trig_sig, time_trig, amp_threshold, num_trials, debugging=False):
    """
    Detects the rising and falling edges of the trigger signals.
    Args:
        trig_sig: Trigger signal
        time_trig: Time vector for the trigger signal
        amp_threshold: Threshold factor for edge detection
        num_trials: Number of trials
        debugging: Boolean for debugging figures
    Returns:
        rise_times, rise_idx, fall_times, fall_idx
    """
    # Normalizing the trigger signal
    trig_sig = trig_sig / np.max(trig_sig)

    # Inverting the trigger signal to find falling edges
    inv_trig = -trig_sig

    # Detecting the edges
    diff_trig = np.diff(trig_sig)
    diff_inv_trig = np.diff(inv_trig)

    # Thresholding the deviations
    diff_trig[diff_trig < amp_threshold] = 0
    diff_inv_trig[diff_inv_trig < amp_threshold] = 0

    # Finding the locations of the peaks
    locs, _ = find_peaks(diff_trig)
    locs_inv, _ = find_peaks(diff_inv_trig)

    # Debugging plots
    if debugging==True:

        # Printing the number of triggers detected and number of trials
        print("\nNum Trigs Detected = {} , inv {}".format(len(locs), len(locs_inv)))
        print("Num Trials in Run = {}".format(num_trials))
        print("====NumTrial should be equal to NumTrigs====\n\n\n")

        # plotting block
        plt.figure()
        plt.plot(trig_sig, 'k', linewidth=1.5)
        plt.plot(diff_trig, '--r', linewidth=1)
        plt.scatter(locs, diff_trig[locs], color='red', marker='o', s=30)
        plt.scatter(locs_inv, diff_inv_trig[locs_inv], color='blue', marker='o', s=30)
        plt.xlabel("Time (index)")
        plt.ylabel("Trigger Signal (black), Diff Trigger (red dashed), Detected triggers (red/blue points)")
        plt.ylim([-1.5, 1.5])
        plt.show()

    # Getting rise and fall times and indexes
    rise_idx = locs
    rise_times = time_trig[rise_idx]

    fall_idx = locs_inv
    fall_times = time_trig[fall_idx]

    return rise_times, rise_idx, fall_times, fall_idx


def segment_emg(experiment, participant_id, block, num_trials=20, amp_threshold=0.5, prestim=1, poststim=2,
                fsample=2148.1481):

    # load participant and block
    fname = path + experiment + '/subj' + participant_id + '/emg/' + experiment + '_' + participant_id + '_' + str(
        block) + '.emg'
    df_emg = pd.read_csv(fname, index_col=0)

    trig_sig = df_emg['trigger'].to_numpy()
    time = df_emg['time'].to_numpy()
    df_emg_clean = df_emg.drop(['trigger', 'time'], axis=1)
    channels = df_emg_clean.columns
    emg = df_emg_clean.to_numpy()

    # detect triggers
    rise_times, rise_idx, fall_times, fall_idx = detect_trig(trig_sig, time, amp_threshold, num_trials)

    segmented_emg = np.zeros((num_trials, emg.shape[-1], np.round((prestim + poststim) * fsample).astype(int)))
    for c, idx in enumerate(rise_idx):
        segmented_emg[c] = emg[idx - np.round(prestim * fsample).astype(int):
                               idx + np.round(poststim * fsample).astype(int)].T

    return segmented_emg, channels


# # Replace 'your_data_file.txt' with the path to your data file
path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path
# path = '/Volumes/Diedrichsen_data$/data/SensoriMotorPrediction/'
# experiment = 'smp0'
# participant_id = '100'
# block = 1
#
# fname = path + experiment + '/subj' + participant_id + '/emg/' + experiment + '_' + participant_id + '_' + str(
#     block) + '.emg'
# df_emg = pd.read_csv(fname)
# trig_sig = df_emg['trigger'].to_numpy()
# time = df_emg['time'].to_numpy()
#
# rise_times, rise_idx, fall_times, fall_idx = detect_trig(trig_sig, time, 0.5, 20, debugging=True)
#
# segmented_emg = segment_emg(experiment, participant_id, block)



