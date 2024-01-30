import sys

import numpy as np
import pandas as pd

from smp0.globals import base_dir
from smp0.utils import sort_cues, f_str_latex
from smp0.visual import make_colors
import matplotlib.pyplot as plt

if __name__ == "__main__":
    datatype = sys.argv[1]

    participants = [100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110]

    file_path = base_dir + f'/smp0/smp0_{datatype}_binned.stat'
    data = pd.read_csv(file_path)
    data = data[data['participant_id'].isin(participants)]

    colors = make_colors(len(data['cue'].unique()))
    cues = sort_cues(data['cue'].unique())

    fig, ax = plt.subplots(len(cues), len(data['stimFinger'].unique()),
                           subplot_kw={'projection': 'polar'}, figsize=(6.4, 9.5))

    # Get unique channels and timepoints
    channels = data['channel'].unique()
    timepoints = data['timepoint'].unique()

    # Angle for each channel
    angles = np.linspace(0, 2 * np.pi, len(channels) + 1, endpoint=True)

    n_participants = len(data['participant_id'].unique())
    n_timepoints = len(data['timepoint'].unique())
    n_stimF = len(data['stimFinger'].unique())
    n_cues = len(cues)

    # Plotting data for each timepoint
    for tp in timepoints:
        for c, cue in enumerate(cues):
            for sF, stimFinger in enumerate(data['stimFinger'].unique()):
                ch_dict = {ch: [] for ch in channels}
                for p, participant_id in enumerate(data['participant_id'].unique()):
                    print(f"cue: {cue}, stimFinger: {stimFinger}, timepoint: {tp}, participant: {participant_id}")
                    subset = data.query('timepoint == @tp and '
                                        'stimFinger == @stimFinger '
                                        'and cue == @cue and '
                                        'participant_id == @participant_id')
                    # subset = subset.sort_values(by='channel', ascending=True)
                    for ch, channel in enumerate(channels):
                        if channel in subset['channel'].unique():
                            ch_dict[channel].append(subset.query('channel == @channel')['Value'].mean())

                av = [np.array(ch_dict[ch]).mean() for ch in channels]
                sem = [np.array(ch_dict[ch]).std() / np.sqrt(len(data['participant_id'].unique())) for ch in channels]
                av += av[:1]  # Repeat the first value to close the plot
                sem += sem[:1]

                ax[c, sF].plot(angles, av, label=f'Timepoint {tp}')

                # Customizations
                ax[c, sF].set_theta_zero_location('N')  # Set 0 degrees at the top
                ax[c, sF].set_theta_direction(-1)  # Clockwise
                ax[c, sF].set_rlabel_position(0)  # Position of radial labels
                if (c == 0) & (sF == 1):
                    ax[c, sF].set_xticks(angles)  # Set ticks for each channel
                    f_ch = list([f_str_latex(ch) for ch in channels]) + ['']
                    ax[c, sF].set_xticklabels(f_ch, fontsize=8)  # Label for each channel
                else:
                    ax[c, sF].set_xticklabels([])
                # ax[c, sF].legend(loc='upper right')  # Legend with timepoints

    plt.show()

    data['stimFinger_cue'] = data['stimFinger'].astype(str) + "," + data['cue']

    # Initialize a dictionary to store correlation matrices
    correlation_matrices = {}

    # Iterate over each combination of timepoint and participant
    corr_mat = np.zeros((n_participants, n_timepoints, int(n_stimF * n_cues - 2), int(n_stimF * n_cues - 2)))
    for (tp, p), group in data.groupby(['timepoint', 'participant_id']):
        pivot_table = group.pivot_table(index='channel', columns='stimFinger_cue', values='Value')
        corr_mat[p-100, tp] = pivot_table.corr().to_numpy()

