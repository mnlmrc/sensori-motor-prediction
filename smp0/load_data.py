# Define a function to read the trial data from a file
import os
import warnings

import pandas as pd
import numpy as np

# Replace 'your_data_file.txt' with the path to your data file
path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path
# path = '/Volumes/Diedrichsen_data$/data/SensoriMotorPrediction/'

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
        fid = open(fname, 'rt')
        dat = pd.read_csv(fid, delimiter='\t', engine='python')

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

def load_emg(experiment, participant_id, block,
             muscle_names=["thumb_flex", "index_flex", "middle_flex", "ring_flex", "pinkie_flex",
                        "thumb_ext", "index_ext", "middle_ext", "ring_ext", "pinkie_ext"],
             trigger_name="trigger"):

    fname = path + experiment + '/subj' + participant_id + '/emg/' + experiment + '_' + participant_id + '_' + str(
        block) + '.csv'

    fid = open(fname, 'rt')
    df = pd.read_csv(fid)

    # Find the columns that contain the muscle names
    muscle_columns = find_muscle_columns(df, muscle_names)

    # Adjust to take the column immediately to the right of the named muscle column for EMG data
    correct_emg_columns = [df.columns[df.columns.get_loc(muscle_columns[muscle]) + 1] for muscle in muscle_names]

    # Find the trigger column and adjust to take the column immediately to its right
    trigger_column = find_muscle_columns(df, [trigger_name])[trigger_name]
    correct_trigger_column = df.columns[df.columns.get_loc(trigger_column) + 1]

    # Extracting EMG data and correct trigger data starting from row 6
    correct_emg_data = df[correct_emg_columns].iloc[6:]
    correct_trigger_data = df[correct_trigger_column].iloc[6:]

    # Renaming the columns to the muscle names
    correct_emg_data.columns = muscle_names

    # Combining EMG data with the corrected trigger data
    combined_data = pd.concat([correct_emg_data, correct_trigger_data], axis=1)
    combined_data.rename(columns={'Unnamed: 3': trigger_name}, inplace=True)

    return combined_data

# # Example usage
# file_path = 'path_to_your_emg_data_file.csv'
# muscle_names = ["thumb_flex", "index_flex", "middle_flex", "ring_flex", "pinkie_flex",
#     "thumb_ext", "index_ext", "middle_ext", "ring_ext", "pinkie_ext"]
# experiment = 'smp0'
# participant_id = '100'
# block = 1
#
# combined_df = extract_emg_and_correct_trigger_data(experiment, participant_id, block, muscle_names)
# print(combined_df.head())


