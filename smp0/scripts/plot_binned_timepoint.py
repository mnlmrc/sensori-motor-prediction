import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from PcmPy.util import est_G_crossval, G_to_dist

import numpy as np

from smp0.globals import base_dir
from smp0.utils import sort_cues, f_str_latex
from smp0.visual import make_colors


if __name__ == "__main__":
    datatype = sys.argv[1]

    participants = [100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110]

    file_path = base_dir + f'/smp0/smp0_{datatype}_binned.stat'
    data = pd.read_csv(file_path)
    data = data[data['participant_id'].isin(participants)]

    colors = make_colors(len(data['cue'].unique()))
    cues = sort_cues(data['cue'].unique())
    palette = {cue: color for cue, color in zip(cues, colors)}

    # compute multivariated distance
    n_participants = len(data['participant_id'].unique())
    n_timepoints = len(data['timepoint'].unique())
    n_stimF = len(data['stimFinger'].unique())
    D = np.zeros((n_participants, n_stimF, n_timepoints, len(cues) - 1, len(cues) - 1))
    Dav = np.zeros((n_participants, n_stimF, n_timepoints))
    for p, participant_id in enumerate(data['participant_id'].unique()):
        n_channels = len(data[data['participant_id'] == int(participant_id)]['channel'].unique())
        for sf, stimF in enumerate(data['stimFinger'].unique()):
            for tp in range(n_timepoints):
                ydata = data[(data['participant_id'] == int(participant_id)) &
                             (data['timepoint'] == tp) &
                             (data['stimFinger'] == stimF)]
                Y = ydata['Value'].to_numpy().reshape(((int(len(ydata) / n_channels)), n_channels))
                cond_vec = ydata['cue'].to_numpy()[::n_channels]
                n_blocks = int(Y.shape[0] / 10)
                blocks = np.arange(n_blocks)
                part_vec = np.repeat(blocks, 10)
                G = est_G_crossval(Y, cond_vec, part_vec)[0]
                dist = G_to_dist(G)
                mask = ~np.eye(dist.shape[0], dtype=bool)
                D[p, sf, tp] = dist
                Dav[p, sf, tp] = dist[mask].mean()

    fig, axs = plt.subplots(len(data['channel'].unique()), len(data['stimFinger'].unique()),
                            figsize=(6, 8), sharex=True, sharey=True)

    for ch, channel in enumerate(data['channel'].unique()):
        for sF, stimFinger in enumerate(data['stimFinger'].unique()):
            subset = data[(data['channel'] == channel) & (data['stimFinger'] == stimFinger)]
            sns.barplot(ax=axs[ch, sF], data=subset, x='timepoint', y='Value', hue='cue',
                        estimator='mean', errorbar='se', palette=palette, legend=None,
                        hue_order=['0%', '25%', '50%', '75%', '100%'], )
            axs[ch, sF].set_ylabel('')
            axs[ch, sF].set_xlabel('')
            axs[ch, sF].tick_params(bottom=False)

            axR = axs[ch, sF].twinx()
            yerr = Dav[:, sF, :].std(axis=0) / np.sqrt(n_participants)
            axR.errorbar(np.linspace(0, 3, 4), Dav[:, sF, :].mean(axis=0),
                         marker='o', color='darkgrey', yerr=yerr)
            axR.set_ylim([-.5, 4])

            if sF == 0:
                axs[ch, sF].spines[['top', 'bottom', 'right']].set_visible(False)
                axR.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
                axR.set_yticks([])
            elif sF == 1:
                axs[ch, sF].spines[['top', 'bottom', 'right', 'left']].set_visible(False)
                axR.spines[['top', 'bottom', 'left']].set_visible(False)
                axs[ch, sF].tick_params(left=False)

        fig.text(.5, axs[ch, 0].get_position().p1[1], f_str_latex(channel), va='top', ha='left')

    legend_handles = [mpatches.Patch(color=color, label=cue) for cue, color in palette.items()]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(legend_handles))
    axs[-1, 0].spines[['bottom']].set_visible(True)
    axs[-1, 1].spines[['bottom']].set_visible(True)
    axs[-1, 0].tick_params(bottom=True)
    axs[-1, 1].tick_params(bottom=True)

    plt.show()

    # fig, axs = plt.subplots(len(cues), len(data['stimFinger'].unique()),
    #                         figsize=(6, 8), sharex=True, sharey=True)
    #
    # for c, cue in enumerate(cues):
    #     for sF, stimFinger in enumerate(data['stimFinger'].unique()):
    #         subset = data[(data['cue'] == cue) & (data['stimFinger'] == stimFinger)]
    #
    #         # Creating a bar plot
    #         ax = sns.barplot(ax=axs[c, sF], data=subset, x='timepoint', y='Value', hue='channel',
    #                          estimator='mean', errorbar='se', legend=None)
