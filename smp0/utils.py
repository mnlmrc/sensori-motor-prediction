import os

import numpy as np
from scipy.signal import firwin


def hp_filter(data, n_ord=None, cutoff=None, fsample=None):
    """
    High-pass filter to remove artifacts from EMG signal
    :param cutoff:
    :param n_ord:
    :param data:
    :param fsample:
    :return:
    """
    numtaps = int(n_ord * fsample / self.cutoff)
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


def save_as_np(data, experiment=None, datatype=None, participant_id=None):
    """

    :param data:
    :param experiment:
    :param datatype:
    :param participant_id:
    :return:
    """
    fname = f"{experiment}_{participant_id}"
    filepath = os.path.join(self.path, self.experiment, f"subj{self.participant_id}", "datatype", fname)
    print(f"Saving participant: {self.participant_id}")
    np.save(filepath, data, allow_pickle=False)


def load_as_np(experiment=None, datatype=None, participant_id=None):
    """

    :param experiment:
    :param datatype:
    :param participant_id:
    :return:
    """

    fname = f"{self.experiment}_{self.participant_id}.npy"
    filepath = os.path.join(self.path, self.experiment, f"subj{self.participant_id}", "emg", fname)
    data = np.load(filepath)

    return data



