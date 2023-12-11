import os
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, firwin, filtfilt, resample

import matplotlib
from sklearn.decomposition import NMF

# from load_data import load_dat, load_participants, load_emg
from smp0.util import vlookup_value

matplotlib.use('MacOSX')


class Smp:
    # Path to data
    path = ('/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My '
            'Drive/UWO/SensoriMotorPrediction/')

    # General parameters
    maxBlocks = 10  # blocks in experiment
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

    # ['thumb_flex', 'index_flex', 'middle_flex', 'ring_flex', 'pinkie_flex', 'thumb_ext',
    #             'index_ext', 'middle_ext', 'ring_ext', 'pinkie_ext']  # approx recorded muscles

    def __init__(self, experiment=None, participant_id=None, amp_threshold=2, prestim=1, poststim=2, cutoff=30,
                 n_ord=4):
        super().__init__(experiment, participant_id)  # Initialize the parent class

        # Parameters for hp filter
        self.cutoff = cutoff  # cutoff frequency
        self.n_ord = n_ord

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

        # check if synergies exists for participant
        fname = f"{self.experiment}_{self.participant_id}_syn.json"
        filepath = os.path.join(self.path, self.experiment, f"subj{self.participant_id}", "emg", fname)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.syn = json.load(f)
            self.W = np.load(filepath[:-5] + 'W.npy')
            self.H = np.load(filepath[:-5] + 'H.npy')
        else:
            self.syn = None  # set to None if it doesn't exist
            self.W = None
            self.H = None

        self.muscle_names = vlookup_value(self.participants,
                                          'participant_id',
                                          f"subj{self.participant_id}",
                                          'muscles').split(',')

    def detect_trig(self, emg_sig, trig_sig, time_trig, debugging=False):

        ########## old trigger detection (subj 100-101)
        # trig_sig = trig_sig / np.max(trig_sig)
        # diff_trig = np.diff(trig_sig)
        # diff_trig[diff_trig < self.amp_threshold] = 0
        # locs, _ = find_peaks(diff_trig)
        ##############################################

        trig_sig[trig_sig < self.amp_threshold] = 0
        trig_sig[trig_sig > self.amp_threshold] = 1

        # Inverting the trigger signal to find falling edges
        # inv_trig = -trig_sig

        # Detecting the edges
        diff_trig = np.diff(trig_sig)

        locs = np.where(diff_trig == 1)[0]

        # Debugging plots
        if debugging:
            # Printing the number of triggers detected and number of trials
            print("\nNum Trigs Detected = {}".format(len(locs)))
            print("Num Trials in Run = {}".format(Emg.ntrials))
            print("====NumTrial should be equal to NumTrigs====\n\n\n")

            # plotting block
            plt.figure()
            plt.plot(emg_sig)
            plt.plot(trig_sig, 'k', linewidth=1.5)
            plt.plot(diff_trig, '--r', linewidth=1)
            plt.scatter(locs, diff_trig[locs], color='red', marker='o', s=30)
            # plt.scatter(locs_inv, diff_trig[locs_inv], color='blue', marker='o', s=30)
            plt.xlabel("Time (index)")
            plt.ylabel("Trigger Signal (black), Diff Trigger (red dashed), Detected triggers (red/blue points)")
            plt.ylim([-1.5, 1.5])
            plt.show()

        # Getting rise and fall times and indexes
        rise_idx = locs
        rise_times = time_trig[rise_idx]

        # fall_idx = locs_inv
        # fall_times = time_trig[fall_idx]

        # Sanity check
        if len(rise_idx) != Emg.ntrials:  # | (len(fall_idx) != Emg.ntrials):
            raise ValueError(f"Wrong number of trials: {len(rise_idx)}")

        return rise_times, rise_idx

    def segment(self, df_emg):
        emg_sig = pd.to_numeric(df_emg['ring_flex']).to_numpy()
        trig_sig = pd.to_numeric(df_emg['trigger']).to_numpy()
        time = pd.to_numeric(df_emg['time']).to_numpy()

        # detect triggers
        _, rise_idx = self.detect_trig(emg_sig, trig_sig, time)

        emg_segmented = np.zeros((len(rise_idx), len(self.muscle_names),
                                  int(Emg.fsample * (self.prestim + self.poststim))))
        for tr, idx in enumerate(rise_idx):
            for m, muscle in enumerate(self.muscle_names):
                emg_segmented[tr, m] = df_emg[muscle][idx - int(self.prestim * Emg.fsample):
                                                      idx + int(self.poststim * Emg.fsample)].to_numpy()

        return emg_segmented

    def hp_filter(self, data, fsample=fsample):

        numtaps = int(self.n_ord * Emg.fsample / self.cutoff)
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

    def load_raw(self, filepath, trigger_name="trigger"):

        # read data from .csv file (Delsys output)
        with open(filepath, 'rt') as fid:
            A = []
            for line in fid:
                # Strip whitespace and newline characters, then split
                split_line = [elem.strip() for elem in line.strip().split(',')]
                A.append(split_line)

        # identify columns with data from each muscle
        muscle_columns = {}
        for muscle in self.muscle_names:
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
            df_out[muscle] = pd.to_numeric(df_raw[muscle_columns[muscle]], errors='coerce').replace('',
                                                                                                    np.nan).dropna()  # add EMG to dataframe

        # High-pass filter and rectify EMG
        for col in df_out.columns:
            df_out[col] = df_out[col]  # convert to floats
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

        blocks = vlookup_value(self.participants, 'participant_id', f"subj{self.participant_id}", 'blocksEMG').split(
            ',')

        # loop through blocks
        for block in blocks:
            # load raw emg data in delsys format
            print(f"Loading emg file - participant: {self.participant_id}, block: {block}")
            fname = f"{self.experiment}_{self.participant_id}_{block}.csv"
            filepath = os.path.join(Emg.path, self.experiment, f"subj{self.participant_id}", 'emg', fname)
            df_emg = self.load_raw(filepath, trigger_name="trigger")

            # segment emg data
            segment = self.segment(df_emg)
            self.emg = segment if self.emg is None else np.concatenate((self.emg, segment), axis=0)

    def save_emg(self):
        fname = f"{self.experiment}_{self.participant_id}"
        filepath = os.path.join(self.path, self.experiment, f"subj{self.participant_id}", "emg", fname)
        print(f"Saving participant: {self.participant_id}")
        np.save(filepath, self.emg, allow_pickle=False)

    def save_syn(self):
        fname = f"{self.experiment}_{self.participant_id}_syn.json"
        filepath = os.path.join(self.path, self.experiment, f"subj{self.participant_id}", "emg", fname)
        print(f"Saving participant: {self.participant_id}")
        np.save(filepath[:-5] + 'H', self.H, allow_pickle=False)
        np.save(filepath[:-5] + 'W', self.W, allow_pickle=False)
        with open(filepath, 'w') as handle:
            json.dump(self.syn, handle)

    def sort_by_stimulated_finger(self, data, finger=None):

        if not finger in self.stimFinger.keys():
            raise ValueError("Unrecognized finger")

        D = self.load_dat(self.path)
        blocks = np.array(
            vlookup_value(self.participants, 'participant_id', f"subj{self.participant_id}", 'blocksEMG').split(
                ',')).astype(int)
        idx = D[(D['stimFinger'] == self.stimFinger[finger]) & (D['BN'].isin(blocks))].index
        idx = idx - (self.ntrials * (self.maxBlocks - len(blocks)))
        emg_finger = data[idx]

        return emg_finger

    def sort_by_stimulated_probability(self, data, finger=None, cue=None):

        if (finger not in self.stimFinger.keys()) or (cue not in self.probCue.keys()):
            raise ValueError("Unrecognized finger")

        D = self.D
        blocks = np.array(
            vlookup_value(self.participants, 'participant_id', f"subj{self.participant_id}", 'blocksEMG').split(
                ',')).astype(int)
        idx = D[(D["stimFinger"] == self.stimFinger[finger]) & (D["chordID"] == self.probCue[cue])].index
        idx = idx - (self.ntrials * (self.maxBlocks - len(blocks)))
        emg_finger = data[idx]

        return emg_finger

    def nnmf_over_time(self, random_state=0, max_iter=500):

        # re init syn
        self.W = None
        self.H = None
        self.syn = None

        X = self.emg.reshape(self.emg.shape[1], self.emg.shape[0] * self.emg.shape[-1])

        prev_r_squared = 1
        r_squared_diff = 1
        r_squared = 0
        n = 0
        W = None
        H = None
        r_squared_values = []  # List to store r_squared values
        while r_squared < 0.8:
            n = n + 1
            print(f"NNMF: using {n} components, last R^2={r_squared}")
            model = NMF(n_components=n, init='random', random_state=random_state, max_iter=max_iter)
            W = model.fit_transform(X)  # synergies
            H = model.components_  # weights
            X_hat = np.dot(W, H)  # reconstructed data
            SST = np.sum((X - np.mean(X)) ** 2)
            SSR = np.sum((X - X_hat) ** 2)
            r_squared = 1 - SSR / SST

            r_squared_values.append(r_squared)  # Store the r_squared value

            r_squared_diff = abs(prev_r_squared - r_squared)
            prev_r_squared = r_squared

            if n == len(self.muscle_names):
                break

        H = H.reshape(n, self.emg.shape[0], self.emg.shape[-1]).swapaxes(0, 1)

        self.W = W
        self.H = H
        self.syn = {
            'r_squared': r_squared.tolist(),
            'n_components': n,
            'random_state': random_state,
            'max_iter': max_iter
        }

        # # Plot the R-squared values
        # plt.plot(r_squared_values)
        # plt.xlabel('Iteration')
        # plt.ylabel('R-squared')
        # plt.title('R-squared values over iterations')
        # plt.show()

    def euclidean_distance_probability(self):

        # Define the parameters for sorting in a list of tuples
        sort_params = [
            ('index', "index 25% - ring 75%"),
            ('index', "index 50% - ring 50%"),
            ('index', "index 75% - ring 25%"),
            ('index', "index 100% - ring 0%"),
            ('ring', "index 75% - ring 25%"),
            ('ring', "index 50% - ring 50%"),
            ('ring', "index 25% - ring 75%"),
            ('ring', "index 0% - ring 100%")
        ]

        emg_sorted = [self.sort_by_stimulated_probability(self.emg, finger=finger, cue=cue).mean(axis=0) for finger, cue
                      in sort_params]

        # Use list comprehension to create emg_sorted
        emg_array = np.array(emg_sorted)  # Convert list to numpy array
        num_conditions = emg_array.shape[0]
        num_timepoints = emg_array.shape[2]

        # Initialize an empty array for distances
        dist = np.zeros((num_conditions, num_conditions, num_timepoints))
        dist_win = np.zeros((num_conditions, num_conditions))

        # Iterate over each timepoint
        for t in range(num_timepoints):
            # Compute the difference in the electrode dimension for each pair of conditions at timepoint t
            for i in range(num_conditions):
                for j in range(num_conditions):
                    dist[i, j, t] = np.linalg.norm(emg_array[i, :, t] - emg_array[j, :, t])
                    dist_win[i, j] = np.linalg.norm(
                        emg_array[i, :, np.where((self.timeS > .8) & (self.timeS < 1.2))[0]] -
                        emg_array[j, :, np.where((self.timeS > .8) & (self.timeS < 1.2))[0]])

        return dist, dist_win, sort_params

# myEmg = Emg('smp0', '102')
# # MyEmg.segment_participant()
# W, H, n, r_squared = myEmg.nnmf_over_time()
