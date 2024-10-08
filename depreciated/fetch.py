import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import resample

import globals as gl
from force import get_path_mov


# def load_participants(experiment):
#     """
#
#     :param experiment:
#     :return:
#     """
#     filepath = os.path.join(gl.base_dir, experiment, "participants.tsv")
#     fid = open(filepath, 'rt')
#     participants = pd.read_csv(fid, delimiter='\t', engine='python', index_col='participant_id')
#
#     return participants


# def load_dat(experiment, session, participant_id):
#
#     path = get_path_mov(experiment, session, participant_id)
#     sn = int(''.join([c for c in participant_id if c.isdigit()]))
#     dat = pd.read_csv(os.path.join(path, f'{experiment}_{sn}.dat'), sep='\t')
#
#     return dat


# def save_npy(data, descriptor, experiment=None, folder=None, participant_id=None):
#     """
#
#     Args:
#         data:
#         descriptor:
#         experiment:
#         folder:
#         participant_id:
#
#     Returns:
#
#     """
#
#     fname = f"{experiment}_{participant_id}"
#     filepath = os.path.join(gl.make_dirs(experiment, folder, participant_id), fname)
#     print(f"Saving data to {filepath}")
#     np.savez(filepath, data_array=data, descriptor=descriptor, allow_pickle=False)
#     print("Data saved!")


# def load_npy(experiment=None, folder=None, participant_id=None):
#     """
#
#     Args:
#         experiment:
#         folder:
#         participant_id:
#
#     Returns:
#
#     """
#
#     fname = f"{experiment}_{participant_id}.npy"
#     filepath = os.path.join(gl.make_dirs(experiment, folder, participant_id), fname)
#     print(f"Loading data from {filepath}")
#     data = np.load(filepath)
#
#     return data


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
    filepath = os.path.join(gl.make_dirs(experiment, "emg", participant_id), fname)

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


def load_mov(filename):
    """
    load .mov file of one block

    :return:
    """

    try:
        with open(filename, 'rt') as fid:
            trial = 0
            A = []
            for line in fid:
                if line.startswith('Trial'):
                    trial_number = int(line.split(' ')[1])
                    trial += 1
                    if trial_number != trial:
                        warnings.warn('Trials out of sequence')
                        trial = trial_number
                    A.append([])
                else:
                    # Convert line to a numpy array of floats and append to the last trial's list
                    data = np.fromstring(line, sep=' ')
                    if A:
                        A[-1].append(data)
                    else:
                        # This handles the case where a data line appears before any 'Trial' line
                        warnings.warn('Data without trial heading detected')
                        A.append([data])

            # Convert all sublists to numpy arrays
            mov = [np.array(trial_data) for trial_data in A]
            # # vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
            # state = [np.array(trial_data) for trial_data in A]

    except IOError as e:
        raise IOError(f"Could not open {filename}") from e

    return mov


