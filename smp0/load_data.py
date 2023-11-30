# Define a function to read the trial data from a file
import os
import warnings

import pandas as pd
import numpy as np

# Replace 'your_data_file.txt' with the path to your data file
path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path
# path = '/Volumes/Diedrichsen_data$/data/SensoriMotorPrediction/'

def count_blocks(experiment, participant_id, extension='.mov'):
    """Count the number of files with a given extension in a directory."""

    directory = path + experiment + '/subj' + participant_id + '/' + extension[1:] + '/'

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
        fid = open(fname, 'rt')
        dat = pd.read_csv(fid, delimiter='\t', engine='python')

    except IOError as e:
        raise IOError(f"Could not open {fname}") from e

    # dat = pd.read_csv(path + experiment + '/subj' + participant_id + '/' + experiment + '_' + participant_id + '.dat', delimiter='\t')

    return dat

def load_emg(experiment='smp0', participant_id='100', block=1):

    fname = path + experiment + '/subj' + participant_id + '/emg/' + experiment + '_' + participant_id + '_' + str(block) + '.csv'

    fid = open(fname, 'rt')
    emg = pd.read_csv(fid, engine='python')

    header = emg[:6]
    emg = emg[6:].to_numpy()

