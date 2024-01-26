import numpy as np
from scipy.signal import firwin, filtfilt

from sklearn.metrics import mean_squared_error

from .fetch import load_dat


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




# def average_condition(Y, Z):
#
#     N, n_channels, n_timepoints = Y.shape
#
#     n_cond = Z.shape[1]
#
#     M = np.zeros((n_cond, n_channels, n_timepoints))
#     for cond in range(n_cond):
#         M[cond, ...] = Y[Z[:, cond]].mean(axis=0)
#
#     return M


def bin_traces(Y, wins, fsample=None, offset=None):
    wins = [(int((offset + win[0]) * fsample), int((offset + win[1]) * fsample)) for win in wins]
    bins = np.array([Y[..., win[0]:win[1]].mean(axis=-1) for win in wins]).transpose((1, 2, 0))

    return bins


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
