# Define a function to read the trial data from a file
import os
import warnings

import chardet
import pandas as pd
import numpy as np


def load_participants(experiment):
    fname = path + experiment + '/' + 'participants.tsv'
    fid = open(fname, 'rt')
    participants = pd.read_csv(fid, delimiter='\t', engine='python')

    return participants


def count_blocks(experiment, participant_id, folder='mov', extension='.mov'):
    """Count the number of files with a given extension in a directory."""

    directory = path + experiment + '/subj' + participant_id + '/' + folder + '/'

    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            count += 1
    return count


def load_mov(experiment, participant_id, block):
    """
    Loads a .mov file, parsed into single trials.
    Checks for consecutive numbering of the trials and warns if trials are missing or out of order.

    Parameters:
    fname : str
        Filename of the .mov file to load.

    Returns:
    A : list
        A list of numpy arrays, each containing the data for one trial.
    """

    try:
        int(participant_id)
        fname = path + experiment + '/subj' + participant_id + '/mov/' + experiment + '_' + participant_id + '_' + block + '.mov'
    except:
        fname = path + experiment + '/' + participant_id + '/mov/' + experiment + '_' + participant_id + '_' + block + '.mov'

    try:
        with open(fname, 'rt') as fid:
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
            rawForce = [np.array(trial_data)[:, 4:9] for trial_data in A]
            vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
            time = [np.array(trial_data)[:, 1:4] for trial_data in A]

    except IOError as e:
        raise IOError(f"Could not open {fname}") from e

    return rawForce, vizForce, time


def load_dat(experiment, participant_id):
    # This function loads the .dat file

    try:
        int(participant_id)
        fname = path + experiment + '/subj' + participant_id + '/' + experiment + '_' + participant_id + '.dat'
    except:
        fname = path + experiment + '/' + participant_id + '/' + experiment + '_' + participant_id + '.dat'

    try:
        # fid = open(fname, 'rt')
        # dat = pd.read_csv(fid, delimiter='\t', engine='python')
        with open(fname, 'rt') as fid:
            A = []
            for line in fid:
                # Strip whitespace and newline characters, then split
                split_line = [elem.strip() for elem in line.strip().split(',')]
                A.append(split_line)

        dat = pd.DataFrame(A)  # get rid of header

    except IOError as e:
        raise IOError(f"Could not open {fname}") from e

    # dat = pd.read_csv(path + experiment + '/subj' + participant_id + '/' + experiment + '_' + participant_id + '.dat', delimiter='\t')

    return dat


def find_muscle_columns(df, muscle_names):
    muscle_columns = {}
    for muscle in muscle_names:
        for col in df.columns:
            if any(df[col].astype(str).str.contains(muscle, na=False)):
                muscle_columns[muscle] = col
                break
    return muscle_columns


def load_emg(experiment, participant_id, block, muscle_names, trigger_name="trigger"):
    fname = path + experiment + '/subj' + participant_id + '/emg/' + experiment + '_' + participant_id + '_' + str(
        block) + '.csv'

    # read data from .csv file (Delsys output)
    with open(fname, 'rt') as fid:
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
        df_out[muscle] = df_raw[muscle_columns[muscle]]  # add EMG to dataframe

    # add trigger column
    for c, col in enumerate(A[3]):
        if trigger_name in col:
            trigger_column = c + 1

    df_out[trigger_name] = df_raw[trigger_column]

    # add time column
    df_out['time'] = df_raw.loc[:, 0]

    return df_out


# Replace 'your_data_file.txt' with the path to your data file
path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path
# path = '/Volumes/Diedrichsen_data$/data/SensoriMotorPrediction/'
