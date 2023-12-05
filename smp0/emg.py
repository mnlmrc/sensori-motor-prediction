import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

import matplotlib

from load_data import load_dat, load_participants, load_emg

matplotlib.use('MacOSX')


def detect_trig(trig_sig, time_trig, amp_threshold=0.4, debugging=False):
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
    if debugging == True:
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

    # sanity check
    if (len(rise_idx) != num_trials) | (len(fall_idx) != num_trials):
        raise ValueError("Wrong number of trials")

    return rise_times, rise_idx, fall_times, fall_idx


def segment_emg(df_emg, prestim=1, poststim=2):
    trig_sig = pd.to_numeric(df_emg['trigger']).to_numpy()
    time = pd.to_numeric(df_emg['time']).to_numpy()
    # df_emg_clean = df_emg.drop(['trigger', 'time'], axis=1)
    timeS = np.linspace(-prestim, poststim, int((prestim + poststim) * fsample))

    # detect triggers
    _, rise_idx, _, _ = detect_trig(trig_sig, time)

    emg_segmented = np.zeros((len(rise_idx), len(muscle_names) + 1, int(fsample * (prestim + poststim))))
    for tr, idx in enumerate(rise_idx):
        for m, muscle in enumerate(muscle_names):
            emg_segmented[tr, m] = df_emg[muscle][idx - np.round(prestim * fsample).astype(int):
                                                            idx + np.round(poststim * fsample).astype(int)].to_numpy()
        emg_segmented[tr, -1] = timeS

    return emg_segmented


def participant(experiment, participant_id, prestim=1, poststim=2):

    # load .dat file to read subj and blocks
    D = load_dat(experiment, participant_id)

    # loop through blocks
    emg = None
    for block in D.BN.unique():

        # load raw emg data in delsys format
        print(f"Loading emg file - participant: {participant_id}, block: {block}")
        emg_file_name = f"{experiment}_{participant_id}_{block}.csv"
        emg_path = os.path.join(path, experiment, f"subj{participant_id}", 'emg', emg_file_name)
        df_emg = load_emg(emg_path, muscle_names=muscle_names, fsample=fsample, trigger_name="trigger")

        # segment emg data
        segment = segment_emg(df_emg, prestim=prestim, poststim=poststim)
        emg = segment if emg is None else np.concatenate((emg, segment), axis=0)

    return emg



# Replace 'your_data_file.txt' with the path to your data file
path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path
num_trials = 20
fsample = 2148.1481
muscle_names = ['thumb_flex', 'index_flex', 'middle_flex', 'ring_flex', 'pinkie_flex', 'thumb_ext', 'index_ext',
                'middle_ext', 'ring_ext', 'pinkie_ext']

# Test participant
# df_emg = participant(experiment='smp0', participant_id='100')

# Test detect_trigger
# emg_file_name = f"smp0_100_2.csv"
# emg_path = os.path.join(path, 'smp0', f"subj100", 'emg', emg_file_name)
# df_emg = load_emg(emg_path, muscle_names=muscle_names, fsample=fsample, trigger_name="trigger")
# trig_sig = pd.to_numeric(df_emg['trigger']).to_numpy()
# time = pd.to_numeric(df_emg['time']).to_numpy()
# detect_trig(trig_sig, time, amp_threshold=0.4, debugging=True)
