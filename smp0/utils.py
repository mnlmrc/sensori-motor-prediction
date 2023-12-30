import warnings
from itertools import product

import numpy as np
from PcmPy import indicator
from scipy.signal import firwin, filtfilt

import smp0.experiment as exp
from smp0.load_and_save import load_npy


def hp_filter(data, n_ord=None, cutoff=None, fsample=None):
    """
    High-pass filter to remove artifacts from EMG signal
    :param cutoff:
    :param n_ord:
    :param data:
    :param fsample:
    :return:
    """
    numtaps = int(n_ord * fsample / cutoff)
    b = firwin(numtaps + 1, cutoff, fs=fsample, pass_zero='highpass')
    filtered_data = filtfilt(b, 1, data)

    return filtered_data


# def centered_moving_average(data, window_size, axis=-1):
#     if window_size % 2 == 0:
#         raise ValueError("Window size should be odd.")
#
#     # Create a sliding window view of the data
#     windowed_data = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size, axis=axis)
#
#     # Compute the mean along the window
#     smoothed_data = np.mean(windowed_data, axis=axis)
#
#     # Since the sliding window view reduces the shape of the array
#     # on both sides, pad the result to match the original data length
#     pad_length = (window_size - 1) // 2
#     smoothed_data = np.pad(smoothed_data, (pad_length, pad_length), mode='edge')
#
#     return smoothed_data
#
#
# def hotelling_t2_test_1_sample(data, baseline):
#     """
#     Perform a one-sample Hotelling's T² test for data in a NumPy array.
#
#     :param data: A NumPy array where each column is a variable and each row is an observation.
#     :param baseline: A baseline mean vector (NumPy array) to compare against.
#     :return: Hotelling's T² statistic and p-value.
#     """
#     n, p = data.shape
#     mean_vector = np.mean(data, axis=0)
#     covariance_matrix = np.cov(data, rowvar=False)
#     difference = mean_vector - baseline
#
#     # Calculate Hotelling's T² statistic
#     t2_stat = n * np.dot(np.dot(difference.T, np.linalg.inv(covariance_matrix)), difference)
#
#     # Transform to F-distribution
#     f_stat = (n - p) / (p * (n - 1)) * t2_stat
#     p_value = f.sf(f_stat, p, n - p)  # sf is the survival function (1 - cdf)
#
#     return t2_stat, p_value
#
#
# def filter_pval_series(pvals, n, threshold=0.05, fsample=None, prestim=None):
#     """
#     Filter segments where p-value is less than a threshold for at least n consecutive samples.
#
#     :param pvals: Array of p-values.
#     :param n: Minimum number of consecutive samples below threshold.
#     :param threshold: Threshold for p-values (default is 0.05).
#     :return: Boolean array where True indicates the start of a segment that meets the criteria.
#     """
#     if n <= 0:
#         raise ValueError("n must be a positive integer")
#
#     # Convert pvals to a boolean array (True if pval < threshold)
#     below_threshold = np.array(pvals) < threshold
#
#     # Initialize an array to store the start of valid segments
#     valid_starts = np.zeros_like(below_threshold, dtype=bool)
#
#     # Iterate over the p-values and check for consecutive runs
#     for i in range(len(pvals) - n + 1):
#         if all(below_threshold[i:i + n]):
#             valid_starts[i] = True
#
#     diff = np.diff(np.concatenate(([0], valid_starts.astype(int), [0])))
#     start_indices = (np.where(diff == 1)[0] / fsample) - prestim
#
#     return valid_starts, start_indices


def Z_condition(conditions):
    """

    :param conditions:
    :return:
    """

    if not all(len(condition) == len(conditions[0]) for condition in conditions):
        raise ValueError("Inconsistent number of trials")

    ntrial = len(conditions[0])
    Z = np.ones(ntrial, dtype=bool)
    for condition in conditions:
        Zi = indicator(condition).astype(bool)
        for _ in range(Z.ndim - 1):
            Zi = Zi[..., np.newaxis, :]
        Z = Z[..., np.newaxis] * Zi

    return Z


# def sort_by_probability(data, Z):
#     c_ord = [4, 0, 3, 1, 2]
#
#     sorted_mean = np.zeros((2, Z.shape[2], data.shape[-2], data.shape[-1]))  # dimord: condition_channel_time
#     # condition = []
#     sorted = [], []
#     for i, c in enumerate(c_ord):
#         sorted[0].append(data[Z[0, :, c]])
#         sorted[1].append(data[Z[1, :, c]])
#         sorted_mean[0, i] = data[Z[0, :, c]].mean(axis=0)
#         sorted_mean[1, i] = data[Z[1, :, c]].mean(axis=0)
#         # condition.append(list(task["cues"].keys())[c])  # wrong order!!!
#
#     return sorted, sorted_mean


# def pool_participants(experiment, conditions_keys=None, channels_key=None, datatype=None):
#
#     MyExp = exp.Experiment(experiment)
#     participants = exp.participants
#     sorted_mean = {}
#     for participant_id in participants:
#
#         # load .npy data of type datatype
#         data = load_npy(experiment, participant_id, datatype)
#
#         # get channel names from participants.tsv
#         channels = MyExp.p_dict[datatype][participant_id][channels_key]
#
#         conditions = []
#         for ck in conditions_keys:
#             conditions.append(MyExp.p_dict[datatype][participant_id][ck])
#
#         Z = Z_condition(conditions)  # the first dimension of Z is always trials
#         for c1 in range(Z.shape[1]):
#             for c2 in range(Z.shape[2]):
#                 for c, ch in enumerate(channels):
#
#                     key1 = list(exp.stimFinger.keys())[c1]
#                     key2 = list(exp.cues.keys())[c2]
#
#                     sorted_mean.setdefault(key1, {}).setdefault(key2, {}).setdefault(ch, [])
#
#                     # Check if the slice is empty before calculating the mean
#                     slice_data = data[Z[:, c1, c2], c]
#                     if not slice_data.size == 0:
#                         mean_value = slice_data.mean(axis=0)
#                         sorted_mean[key1][key2][ch].append(mean_value)
#
#     return sorted_mean

def pool_participants(experiment, conditions_keys=None, channels_name=None, datatype=None):
    MyExp = exp.Experiment(experiment)
    participants = exp.participants
    sorted_mean = {}
    for participant_id in participants:
        # Load .npy data of type datatype
        data = load_npy(experiment, participant_id, datatype)

        # Get channel names from participants.tsv
        channels = MyExp.p_dict[datatype][participant_id][channels_name]

        # creat list of condition vectors
        conditions = [MyExp.p_dict[datatype][participant_id][ck] for ck in conditions_keys]
        Z = Z_condition(conditions)

        # Generate all combinations of indices for the dimensions of Z (excluding the first one)
        indices = [range(dim_size) for dim_size in Z.shape[1:]]  # the first dimension of Z is always trials

        # for index_tuple in product(*indices):
        keys = [list(exp.conditions[ck].keys())
                for ck in conditions_keys]

        # Build the nested dictionary structure
        # current_dict = sorted_mean
        for combination, index_tuple in zip(product(*keys), product(*indices)):
            # current_dict = current_dict.setdefault(k, {})

            # Navigate or create the nested dictionary structure
            current_dict = sorted_mean
            for key in combination[:-1]:  # Exclude the last key for now
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]

            # The last key in the combination will be used for the channels
            last_key = combination[-1]
            if last_key not in current_dict:
                current_dict[last_key] = {ch: [] for ch in channels}

            for ch in channels:
                if ch not in current_dict[last_key]:
                    current_dict[last_key][ch] = []

            for c, ch in enumerate(channels):
                # Construct the full index including the channel
                z_index = Z[(slice(None),) + index_tuple].astype(bool)

                # Check if the slice is empty before calculating the mean
                slice_data = data[z_index, c]
                if slice_data.size != 0:
                    mean_value = slice_data.mean(axis=0)
                    current_dict[last_key][ch].append(mean_value)

    return sorted_mean


# Example usage
# experiment = "smp0"
# conditions_keys = ["stimFinger", "cues"]  # Update as needed
# channels_key = "muscle_names"
# datatype = "emg"
# data = pool_participants(experiment, conditions_keys, channels_key, datatype)

# def detect_response_latency(data, threshold=None, fsample=None):
#     return np.where(data > threshold)[0][0] / fsample

# MyExp = Exp()
