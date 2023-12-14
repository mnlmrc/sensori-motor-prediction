import pandas as pd
from scipy.signal import resample


def load_delsys(filepath, muscle_names=None, trigger_name=None):
    """returns a pandas DataFrame with the raw EMG data recorded using the Delsys system

    :param filepath: path to the .csv data exported from Delsys Trigno software
    :param muscle_names: 
    :param trigger_name:
    :return:
    """
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
        df_out[muscle] = pd.to_numeric(df_raw[muscle_columns[muscle]], errors='coerce').replace('',
                                                                                                np.nan).dropna()  # add EMG to dataframe

    # # High-pass filter and rectify EMG
    # for col in df_out.columns:
    #     df_out[col] = df_out[col]  # convert to floats
    #     df_out[col] = hp_filter(df_out[col])
    #     df_out[col] = df_out[col].abs()  # Rectify

    # add trigger column
    trigger_column = None
    for c, col in enumerate(A[3]):
        if trigger_name in col:
            trigger_column = c + 1

    # try:
    trigger = df_raw[trigger_column]
    trigger = resample(trigger.values, len(df_out))
    df_out[trigger_name] = trigger
    # except:
    #     raise ValueError("Trigger not found")

    # add time column
    df_out['time'] = df_raw.loc[:, 0]

    return df_out