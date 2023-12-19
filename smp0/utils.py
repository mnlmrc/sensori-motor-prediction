import numpy as np
from PcmPy import indicator
from scipy.signal import firwin, filtfilt
from scipy.stats import f

# from smp0.info import task, timeS
from smp0.load_and_save import load_npy, load_participants, load_dat


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


def vlookup_value(df, search_column, search_value, target_column):
    """
    Returns the value from 'target_column' where 'search_column' equals 'search_value'.

    :param df: Pandas DataFrame to search in.
    :param search_column: Column name in which to search for 'search_value'.
    :param search_value: Value to search for in the 'search_column'.
    :param target_column: Column from which to return the value.
    :return: Value from 'target_column' or None if 'search_value' is not found.
    """
    matching_rows = df[df[search_column] == search_value]
    if not matching_rows.empty:
        return matching_rows.iloc[0][target_column]
    else:
        return None


def centered_moving_average(data, window_size, axis=-1):
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd.")

    # Create a sliding window view of the data
    windowed_data = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size, axis=axis)

    # Compute the mean along the window
    smoothed_data = np.mean(windowed_data, axis=axis)

    # Since the sliding window view reduces the shape of the array
    # on both sides, pad the result to match the original data length
    pad_length = (window_size - 1) // 2
    smoothed_data = np.pad(smoothed_data, (pad_length, pad_length), mode='edge')

    return smoothed_data


def hotelling_t2_test_1_sample(data, baseline):
    """
    Perform a one-sample Hotelling's T² test for data in a NumPy array.

    :param data: A NumPy array where each column is a variable and each row is an observation.
    :param baseline: A baseline mean vector (NumPy array) to compare against.
    :return: Hotelling's T² statistic and p-value.
    """
    n, p = data.shape
    mean_vector = np.mean(data, axis=0)
    covariance_matrix = np.cov(data, rowvar=False)
    difference = mean_vector - baseline

    # Calculate Hotelling's T² statistic
    t2_stat = n * np.dot(np.dot(difference.T, np.linalg.inv(covariance_matrix)), difference)

    # Transform to F-distribution
    f_stat = (n - p) / (p * (n - 1)) * t2_stat
    p_value = f.sf(f_stat, p, n - p)  # sf is the survival function (1 - cdf)

    return t2_stat, p_value


def filter_pval_series(pvals, n, threshold=0.05, fsample=None, prestim=None):
    """
    Filter segments where p-value is less than a threshold for at least n consecutive samples.

    :param pvals: Array of p-values.
    :param n: Minimum number of consecutive samples below threshold.
    :param threshold: Threshold for p-values (default is 0.05).
    :return: Boolean array where True indicates the start of a segment that meets the criteria.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Convert pvals to a boolean array (True if pval < threshold)
    below_threshold = np.array(pvals) < threshold

    # Initialize an array to store the start of valid segments
    valid_starts = np.zeros_like(below_threshold, dtype=bool)

    # Iterate over the p-values and check for consecutive runs
    for i in range(len(pvals) - n + 1):
        if all(below_threshold[i:i + n]):
            valid_starts[i] = True

    diff = np.diff(np.concatenate(([0], valid_starts.astype(int), [0])))
    start_indices = (np.where(diff == 1)[0] / fsample) - prestim

    return valid_starts, start_indices


# def Z_probability(d, stimFinger=None):
#     """
#
#     :param d:
#     :param blocks:
#     :param stimFinger:
#     :return:
#     """
#     # if stimFinger not in info.task["stimFinger"]:
#     #     raise ValueError("Unrecognized finger")
#
#     idxf = list(task["stimFinger"].keys()).index(stimFinger)
#
#     Zp = indicator(d.chordID).astype(bool)
#     Zf = indicator(d.stimFinger)[:, idxf].astype(bool)
#
#     Zp = Zp * Zf.reshape(-1, 1)
#
#     return Zp


# def sort_by_probability(data, Z):
#     c_ord = [4, 0, 3, 1, 2]
#
#     sorted_mean = np.zeros((Z.shape[1], data.shape[1], data.shape[2]))  # dimord: condition_channel_time
#     condition = []
#     sorted = []
#     for i, c in enumerate(c_ord):
#         sorted.append(data[Z[:, c]])
#         sorted_mean[i] = data[Z[:, c]].mean(axis=0)
#         condition.append(list(task["cues"].keys())[c])  # wrong order!!!
#
#     return sorted, sorted_mean, condition


# def pool_participants(experiment, participants, datatype=None):
#     info_p = load_participants(experiment)
#
#     n_participants = len(participants)
#     n_stimF = len(task["stimFinger"])
#     n_cues = len(task["cues"])
#     n_timep = len(timeS[datatype])
#
#     n_ch = None
#     if datatype == 'emg':
#         n_ch = 11
#     elif datatype == 'mov':
#         n_ch = 5
#
#     data_p = np.zeros((n_participants,
#                        n_stimF,
#                        n_cues,
#                        n_ch,
#                        n_timep))
#     for c, participant_id in enumerate(participants):
#
#         print(participant_id)
#
#         d = load_dat(experiment,
#                      participant_id)
#         blocks = vlookup_value(info_p,
#                                'participant_id',
#                                f"subj{participant_id}",
#                                f"blocks_{datatype}").split(",")
#         blocks = [int(block) for block in blocks]
#         d = d[d.BN.isin(blocks)]
#
#         data = load_npy(experiment=experiment,
#                         participant_id=participant_id,
#                         datatype=datatype)
#
#         Zi = Z_probability(d, stimFinger="index")
#         Zr = Z_probability(d, stimFinger="ring")
#         _, sorted_mean_i, condition = sort_by_probability(data, Zi)
#         _, sorted_mean_r, _ = sort_by_probability(data, Zr)
#         sorted_mean = np.stack([sorted_mean_i, sorted_mean_r], axis=0)
#         if datatype == 'emg':
#             muscle_names = vlookup_value(info_p,
#                                          'participant_id',
#                                          f"subj{participant_id}",
#                                          'muscle_names').split(",")
#             for i, muscle in enumerate(muscle_names):
#                 m = task["Muscles"].index(muscle)
#                 data_p[c, :, :, m, :] = sorted_mean[..., i, :]
#         elif datatype == 'mov':
#             data_p[c] = np.stack([sorted_mean_i, sorted_mean_r], axis=0)
#
#     return data_p


def detect_response_latency(data, threshold=None, fsample=None):
    return np.where(data > threshold)[0][0] * fsample








