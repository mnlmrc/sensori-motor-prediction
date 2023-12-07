import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, firwin, filtfilt, resample

import matplotlib

from load_data import load_dat, load_participants, load_emg

matplotlib.use('MacOSX')


class Smp:
    # Path to data
    path = ('/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My '
            'Drive/UWO/SensoriMotorPrediction/')

    # General parameters
    nblocks = 10  # blocks in experiment
    ntrials = 20  # trials per block

    # Map trial information
    stimFinger = {
        "index": 91999,
        "ring": 99919
    }

    probCue = {
        "index 0% - ring 100%": 93,
        "index 25% - ring 75%": 12,
        "index 50% - ring 50%": 44,
        "index 75% - ring 25%": 21,
        "index 100% - ring 0%": 39
    }

    def __init__(self, experiment=None, participant_id=None):
        self.experiment = experiment
        self.participant_id = participant_id
        self.D = self.load_dat(Smp.path)
        self.participants = self.load_participant(Smp.path)

    def load_participant(self, path):
        filepath = os.path.join(path, self.experiment, "participants.tsv")
        fid = open(filepath, 'rt')
        participants = pd.read_csv(fid, delimiter='\t', engine='python')

        return participants

    def load_dat(self, path):
        fname = f"{self.experiment}_{self.participant_id}.dat"
        filepath = os.path.join(path, self.experiment, f"subj{self.participant_id}", fname)

        try:
            fid = open(filepath, 'rt')
            D = pd.read_csv(fid, delimiter='\t', engine='python')
        except IOError as e:
            raise IOError(f"Could not open {filepath}") from e

        return D


class Emg(Smp):
    # EMG general parameters
    fsample = 2148.1481  # sampling rate EMG
    muscle_names = ['thumb_flex', 'index_flex', 'middle_flex', 'ring_flex', 'pinkie_flex', 'thumb_ext',
                    'index_ext', 'middle_ext', 'ring_ext', 'pinkie_ext']  # approx recorded muscles

    def __init__(self, experiment=None, participant_id=None, amp_threshold=2, prestim=1, poststim=2, cutoff=30,
                 nOrd=4):
        super().__init__(experiment, participant_id)  # Initialize the parent class

        # Parameters for hp filter
        self.cutoff = cutoff  # cutoff frequency
        self.nOrd = nOrd

        # EMG segmentation parameters
        self.amp_threshold = amp_threshold  # threshold for trigger detection
        self.prestim = prestim  # pre trigger time
        self.poststim = poststim  # post trigger time

        # Time axis for segmented data
        self.timeS = np.linspace(self.prestim * -1, self.poststim,
                                 int((self.prestim + self.poststim) * Emg.fsample))

        # check if segmented emg exists for participant
        fname = f"{self.experiment}_{self.participant_id}.npy"
        filepath = os.path.join(self.path, self.experiment, f"subj{self.participant_id}", "emg", fname)
        if os.path.exists(filepath):
            self.emg = np.load(filepath)  # load segmented emg if it exists
        else:
            self.emg = None  # set to None if it doesn't exist

    def detect_trig(self, trig_sig, time_trig, debugging=True):

        # Normalizing the trigger signal
        # trig_sig = trig_sig / np.max(trig_sig)
        trig_sig[trig_sig < self.amp_threshold] = 0
        trig_sig[trig_sig > self.amp_threshold] = 1

        # Inverting the trigger signal to find falling edges
        inv_trig = -trig_sig

        # Detecting the edges
        diff_trig = np.diff(trig_sig)

        locs = np.where(diff_trig == 1)[0]
        locs_inv = np.where(diff_trig == -1)[0]
        # diff_inv_trig = np.diff(inv_trig)

        # # Thresholding the deviations
        # diff_trig[diff_trig < self.amp_threshold] = 0
        # diff_inv_trig[diff_inv_trig < self.amp_threshold] = 0
        #
        # # Finding the locations of the peaks
        # locs, _ = find_peaks(diff_trig)
        # locs_inv, _ = find_peaks(diff_inv_trig)

        # Debugging plots
        if debugging:
            # Printing the number of triggers detected and number of trials
            print("\nNum Trigs Detected = {} , inv {}".format(len(locs), len(locs_inv)))
            print("Num Trials in Run = {}".format(Emg.ntrials))
            print("====NumTrial should be equal to NumTrigs====\n\n\n")

            # plotting block
            plt.figure()
            plt.plot(trig_sig, 'k', linewidth=1.5)
            plt.plot(diff_trig, '--r', linewidth=1)
            plt.scatter(locs, diff_trig[locs], color='red', marker='o', s=30)
            plt.scatter(locs_inv, diff_trig[locs_inv], color='blue', marker='o', s=30)
            plt.xlabel("Time (index)")
            plt.ylabel("Trigger Signal (black), Diff Trigger (red dashed), Detected triggers (red/blue points)")
            plt.ylim([-1.5, 1.5])
            plt.show()

        # Getting rise and fall times and indexes
        rise_idx = locs
        rise_times = time_trig[rise_idx]

        fall_idx = locs_inv
        fall_times = time_trig[fall_idx]

        # Sanity check
        if (len(rise_idx) != Emg.ntrials) | (len(fall_idx) != Emg.ntrials):
            raise ValueError("Wrong number of trials")

        return rise_times, rise_idx, fall_times, fall_idx

    def segment(self, df_emg):
        trig_sig = pd.to_numeric(df_emg['trigger']).to_numpy()
        time = pd.to_numeric(df_emg['time']).to_numpy()

        # detect triggers
        _, rise_idx, _, _ = self.detect_trig(trig_sig, time)

        emg_segmented = np.zeros((len(rise_idx), len(Emg.muscle_names),
                                  int(Emg.fsample * (self.prestim + self.poststim))))
        for tr, idx in enumerate(rise_idx):
            for m, muscle in enumerate(Emg.muscle_names):
                emg_segmented[tr, m] = df_emg[muscle][idx - int(self.prestim * Emg.fsample):
                                                      idx + int(self.poststim * Emg.fsample)].to_numpy()

        return emg_segmented

    def hp_filter(self, data, fsample=fsample):

        numtaps = int(self.nOrd * Emg.fsample / self.cutoff)
        b = firwin(numtaps + 1, self.cutoff, fs=fsample, pass_zero='highpass')
        filtered_data = filtfilt(b, 1, data)

        return filtered_data

    # def resample_trigger(self, vector, original_freq=2222.2222, new_freq=fsample):
    #     """
    #     Resamples the trigger channel from original_freq to new_freq.
    #     This is because Delsys records trigger and EMG at different frequencies
    #
    #     Parameters:
    #     vector (numpy array): The input signal to resample.
    #     original_freq (int): The original sampling frequency of the vector.
    #     new_freq (int): The desired new sampling frequency.
    #
    #     Returns:
    #     numpy array: The resampled signal.
    #     """
    #     # Calculate the number of samples in the resampled vector
    #     original_len = len(vector)
    #     new_len = int(original_len * new_freq / original_freq)
    #
    #     # Use scipy's resample function to resample the vector
    #     resampled_vector = resample(vector, new_len)
    #
    #     return resampled_vector

    def load_raw(self, emg_path, trigger_name="trigger"):

        # read data from .csv file (Delsys output)
        with open(emg_path, 'rt') as fid:
            A = []
            for line in fid:
                # Strip whitespace and newline characters, then split
                split_line = [elem.strip() for elem in line.strip().split(',')]
                A.append(split_line)

        # identify columns with data from each muscle
        muscle_columns = {}
        for muscle in Emg.muscle_names:
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
            df_out[muscle] = df_raw[muscle_columns[muscle]]  # add EMG to dataframe

        # High-pass filter and rectify EMG
        for col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')  # convert to floats
            df_out[col] = self.hp_filter(df_out[col])
            df_out[col] = df_out[col].abs()  # Rectify

        # add trigger column
        trigger_column = None
        for c, col in enumerate(A[3]):
            if trigger_name in col:
                trigger_column = c + 1

        # try:
        trigger = df_raw[trigger_column]
        trigger = resample(trigger.values, len(df_out))
        df_out[trigger_name] = trigger
        # except:
        #     raise ValueError("Trigger not found")

        # add time column
        df_out['time'] = df_raw.loc[:, 0]

        return df_out

    def segment_participant(self):

        # set emg to None to segment (again) participant
        self.emg = None

        # loop through blocks
        for block in self.D.BN.unique():
            # load raw emg data in delsys format
            print(f"Loading emg file - participant: {self.participant_id}, block: {block}")
            fname = f"{self.experiment}_{self.participant_id}_{block}.csv"
            filepath = os.path.join(Emg.path, self.experiment, f"subj{self.participant_id}", 'emg', fname)
            df_emg = self.load_raw(filepath, trigger_name="trigger")

            # segment emg data
            segment = self.segment(df_emg)
            self.emg = segment if self.emg is None else np.concatenate((self.emg, segment), axis=0)



    def save_segmented(self):
        fname = f"{self.experiment}_{self.participant_id}"
        filepath = os.path.join(self.path, self.experiment, f"subj{self.participant_id}", "emg", fname)
        np.save(filepath, self.emg, allow_pickle=False)


    def sort_by_stimulated_finger(self, finger=None):

        if not finger in self.stimFinger.keys():
            raise ValueError("Unrecognized finger")

        D = self.load_dat(self.path)
        idx = D[D['stimFinger'] == self.stimFinger[finger]].index
        emg_finger = self.emg[idx]

        return emg_finger

    def sort_by_stimulated_probability(self, finger=None, cue=None):

        if (finger not in self.stimFinger.keys()) or (cue not in self.probCue.keys()):
            raise ValueError("Unrecognized finger")

        D = self.D
        idx = D[(D["stimFinger"] == self.stimFinger[finger]) and (D["chordID"] == self.probCue[cue])].index
        emg_finger = self.emg[idx]

        return emg_finger


# MyEmg = Emg('smp0', '102')
# MyEmg.segment_participant()

