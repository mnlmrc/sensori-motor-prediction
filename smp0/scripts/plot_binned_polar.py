import sys
from itertools import product

import numpy as np
import pandas as pd

from smp0.globals import base_dir
from smp0.synergies import nnmf, assign_synergy
from smp0.utils import sort_cues, f_str_latex
from smp0.visual import make_colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

if __name__ == "__main__":
    datatype = sys.argv[1]

    participants = [100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110]

    file_path = base_dir + f"/smp0/smp0_{datatype}_binned.stat"
    data = pd.read_csv(file_path)
    data = data[data['participant_id'].isin(participants)]

    channels = data['channel'].unique()
    timepoints = data['timepoint'].unique()
    stimFingers = data['stimFinger'].unique()
    cues = ['0%', '25%', '50%', '75%', '100%']

    n_participants = len(participants)

    colors = make_colors(len(timepoints), ecol=('green', 'orange'))

    components = ['syn1', 'syn2']
    syn = {(tp, comp, participant_id): []
           for tp, comp, participant_id in product(timepoints, components, participants)}
    for tp, participant_id in product(timepoints, participants):
        pdata = data[(data['participant_id'] == participant_id) & (data['timepoint'] == tp)]
        pchannels = pdata['channel'].unique().tolist()
        n_pchannels = len(pchannels)

        if len(pdata) % n_pchannels == 0:
            X = pdata['Value'].to_numpy().reshape((-1, n_pchannels))
            W, H, R_squared = nnmf(X, n_components=2)

            # Custom logic for H_pred
            H_pred = np.zeros((2, n_pchannels))
            H_pred[0, 1] = 1  # Example logic for syn1
            H_pred[1, 3] = 1  # Example logic for syn2

            _, H = assign_synergy(W, H, H_pred)
            # nH = H / np.linalg.norm(H, axis=1, keepdims=True)

            for channel in channels:
                if channel in pchannels:
                    channel_index = pchannels.index(channel)
                    syn[(tp, 'syn1', participant_id)].append(H[0, channel_index])
                    syn[(tp, 'syn2', participant_id)].append(H[1, channel_index])
                else:
                    syn[(tp, 'syn1', participant_id)].append(np.nan)
                    syn[(tp, 'syn2', participant_id)].append(np.nan)
        else:
            raise ValueError(
                f"Data length for participant {participant_id}, timepoint {tp} is not divisible by number of channels")

    syns = np.zeros((len(timepoints), 2, len(participants), len(channels)))
    for tp, participant_id in product(timepoints, participants):
        angles = np.linspace(0, 2 * np.pi, len(channels) + 1)
        syns[tp, 0, participant_id - 100] = np.array(syn[(tp, 'syn1', participant_id)])
        syns[tp, 1, participant_id - 100] = syn[(tp, 'syn2', participant_id)]

    mval = data.groupby(['timepoint', 'stimFinger', 'cue', 'participant_id', 'channel']).agg(
        {'Value': 'mean'}).reset_index()

    # Simplify with dictionary comprehensions and list comprehensions
    plot_data = {
        (tp, cue, stimF): {
            'av': [
                mval[(mval['timepoint'] == tp) &
                     (mval['stimFinger'] == stimF) &
                     (mval['cue'] == cue) &
                     (mval['channel'] == ch)]['Value'].mean()
                for ch in channels
            ],
            'sem': [
                mval[(mval['timepoint'] == tp) &
                     (mval['stimFinger'] == stimF) &
                     (mval['cue'] == cue) &
                     (mval['channel'] == ch)]['Value'].std() / np.sqrt(n_participants)
                for ch in channels
            ]
        }
        for tp in timepoints
        for cue in cues
        for stimF in stimFingers
    }

    fig, axs = plt.subplots(len(cues), len(data['stimFinger'].unique()),
                            subplot_kw={'projection': 'polar'}, figsize=(6.4, 9.5))

    angles = np.linspace(0, 2 * np.pi, len(channels) + 1)
    labels = ['pre', 'SLR', 'LLR', 'Vol']

    for tp in timepoints:
        for c, cue in enumerate(cues):
            for sF, stimFinger in enumerate(stimFingers):
                av = plot_data[(tp, cue, stimFinger)]['av']
                sem = plot_data[(tp, cue, stimFinger)]['sem']
                av.append(av[0])
                sem.append(sem[0])
                av = np.array(av)
                sem = np.array(sem)

                axs[c, sF].plot(angles, av, color=colors[tp])
                axs[c, sF].fill_between(angles, av - sem, av + sem,
                                        color=colors[tp], alpha=0.2)

                axs[c, sF].set_theta_zero_location('N')  # Set 0 degrees at the top
                axs[c, sF].set_theta_direction(-1)  # Clockwise
                axs[c, sF].set_rlabel_position(0)  # Position of radial labels

                axs[c, sF].set_xticks(angles)  # Set ticks for each channel
                f_channels = [f_str_latex(ch) for ch in channels] + ['']
                axs[c, sF].set_xticklabels(f_channels, fontsize=8)  # Label for each channel
                # else:
                #     axs[c, sF].set_xticklabels([])

    for c, cue in enumerate(cues):
        fig.text(.5, axs[c, 0].get_position().p1[1], cue, va='top', ha='left')

    palette = {label: color for label, color in zip(labels, colors)}
    lh = [mlines.Line2D([], [], color=color, label=cue) for cue, color in palette.items()]
    fig.legend(handles=lh, loc='lower right', ncol=1)

    # fig2, axs2 = plt.subplots()

    for tp in timepoints:
        angles = np.linspace(0, 2 * np.pi, len(channels) + 1)

        # index
        patternI = list(np.nanmean(syns[tp, 0], axis=0))
        patternI.append(patternI[0])
        axs[0, 0].plot(angles, patternI, color=colors[tp])
        axs[0, 0].set_theta_zero_location('N')  # Set 0 degrees at the top
        axs[0, 0].set_theta_direction(-1)  # Clockwise
        axs[0, 0].set_rlabel_position(0)  # Position of radial labels
        axs[0, 0].set_yticks([])

        # ring
        patternR = list(np.nanmean(syns[tp, 1], axis=0))
        patternR.append(patternR[0])
        axs[-1, -1].plot(angles, patternR, color=colors[tp])
        axs[-1, -1].set_theta_zero_location('N')  # Set 0 degrees at the top
        axs[-1, -1].set_theta_direction(-1)  # Clockwise
        axs[-1, -1].set_rlabel_position(0)  # Position of radial labels
        axs[-1, -1].set_yticks([])

        # axs2.scatter(patternI, patternR, color=colors[tp])

        plt.show()

    # axs[0, 0].set_visible(False)
    # axs[-1, -1].set_visible(False)

    # fig.tight_layout()



