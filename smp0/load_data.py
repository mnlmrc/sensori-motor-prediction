# Define a function to read the trial data from a file
import pandas as pd
import numpy as np

# Replace 'your_data_file.txt' with the path to your data file
path = '/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/'

experiment = 'smp0'
subject = 'clamped'
block = '01'

def load_mov(experiment, subject, block):

    # This function loads the .mov file

    df_mov = pd.read_csv(path + experiment + '/' + experiment + '_' + subject + '_' + block + '.mov', delimiter='\t', header=None)
    df_mov = df_mov.dropna()

    trials = df_mov.groupby(df_mov.columns[0])

    # Now create a list of NumPy arrays, excluding the first column which is the grouping column
    rawForce = [group.iloc[:, 4:9].to_numpy() for _, group in trials]
    vizForce = [group.iloc[:, 9:].to_numpy() for _, group in trials]
    time = [group.iloc[:, 1:4].to_numpy() for _, group in trials]

    return rawForce, vizForce, time


def load_dat(experiment, subject):

    # This function loads the .dat file

    dat = pd.read_csv(path + experiment + '/' + experiment + '_' + subject + '.dat', delimiter='\t')

    return dat






