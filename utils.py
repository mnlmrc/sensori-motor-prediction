# from PcmPy import indicator
from scipy.signal import firwin, filtfilt


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


def detect_response_latency(data, threshold=None, fsample=None):
    return np.where(data > threshold)[0][0] / fsample


def sort_by_condition(Y, Z):
    meas = Y.measurements

    n_cond = Z.shape[1]

    Sorted = list()
    for cond in range(n_cond):
        Sorted.append(meas[Z[:, cond]])

    return Sorted


def bin_traces(Y, wins, fsample=None, offset=None):
    wins = [(int((offset + win[0]) * fsample), int((offset + win[1]) * fsample)) for win in wins]
    bins = np.array([Y[..., win[0]:win[1]].mean(axis=-1) for win in wins]).transpose((1, 2, 0))

    return bins


import numpy as np


def av_within_participant(Y, Z, cond_name=None):
    n_cond = Z.shape[1]
    if Y.ndim == 3:
        N, n_channels, n_timepoints = Y.shape
        M = np.zeros((n_cond, n_channels, n_timepoints))
        SD = np.zeros((n_cond, n_channels, n_timepoints))
    elif Y.ndim == 2:
        N, n_channels = Y.shape
        M = np.zeros((n_cond, n_channels))
        SD = np.zeros((n_cond, n_channels))
    else:
        M = None
        SD = None

    for cond in range(n_cond):
        M[cond, ...] = Y[Z[:, cond]].mean(axis=0)
        SD[cond, ...] = Y[Z[:, cond]].std(axis=0)

    if cond_name is None:
        return M, SD
    else:
        return M, SD, cond_name


# def av_across_participants(channels, data):
#     ch_dict = {ch: [] for ch in channels}
#     N = len(data)
#
#     for p_data in data:
#         Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
#         M, _ = av_within_participant(p_data.measurements, Z)
#
#         for ch in channels:
#             if ch in p_data.channel_descriptors['channels']:
#                 ch_index = p_data.channel_descriptors['channels'].index(ch)
#                 ch_dict[ch].append(M[:, ch_index])
#
#     av, sd, sem = {}, {}, {}
#     for ch in channels:
#         ch_data = np.array(ch_dict[ch])
#         ch_dict[ch] = ch_data
#
#         if ch_data.ndim == 3:
#             av[ch] = np.mean(ch_data, axis=0)
#             sd[ch] = np.std(ch_data, axis=0)
#             sem[ch] = (sd[ch] / np.sqrt(N))
#         else:
#             av[ch] = np.mean(ch_data, axis=0)
#             sd[ch] = np.std(ch_data, axis=0)
#             sem[ch] = (sd[ch] / np.sqrt(N))
#
#     return av, sd, sem, ch_dict


def split_column_df(df, new_cols, old_col):
    # Split the 'Combined' column into two new columns
    df[new_cols] = df[old_col].str.split(',', expand=True)
    del df[old_col]
    return df


def remap_chordID(df):
    remapped_dataframes = {}
    mapping_dict = {93: 0, 12: 25, 44: 50, 21: 75, 39: 100}

    # Read the txt file into a dataframe
    try:
        # df = load_dat(experiment, participant_id)
        # Remap the 'chordID' column
        df['cues'] = df['chordID'].map(mapping_dict)
        remapped_dataframes = df
    except Exception as e:
        print(f"An error occurred")

    return remapped_dataframes


def f_str_latex(txt):
    parts = txt.split('_')
    if len(parts) == 2:
        return f"${parts[0]}_{{{parts[1]}}}$"
    else:
        return txt


def sort_cues(cue_list):
    # Convert to integers (or floats) by removing the '%' sign and sorting
    sorted_cues = sorted([int(cue.strip('%')) for cue in cue_list])

    # Convert back to string with '%' sign
    sorted_cues = [f"{cue}%" for cue in sorted_cues]

    return sorted_cues