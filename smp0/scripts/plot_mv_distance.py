import sys

import numpy as np
import pandas as pd

from smp0.globals import base_dir
from smp0.utils import sort_cues
from smp0.visual import make_colors
from PcmPy.util import est_G_crossval, G_to_dist, indicator

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
    palette = {cue: color for cue, color in zip(cues, colors)}

    # compute multivariated distance
    n_participants = len(data['participant_id'].unique())
    n_timepoints = len(data['timepoint'].unique())
    n_stimF = len(data['stimFinger'].unique())
    D = np.zeros((n_participants, n_timepoints, len(cues)*n_stimF - 2, len(cues)*n_stimF - 2))
    Dav = np.zeros((n_participants, n_timepoints))
    for p, participant_id in enumerate(data['participant_id'].unique()):
        n_channels = len(data[data['participant_id'] == int(participant_id)]['channel'].unique())
        for sf, stimF in enumerate(data['stimFinger'].unique()):
            for tp in range(n_timepoints):
                ydata = data[(data['participant_id'] == int(participant_id)) &
                             (data['timepoint'] == tp)]
                Y = ydata['Value'].to_numpy().reshape(((int(len(ydata) / n_channels)), n_channels))
                ydata['cue'][ydata['cue'] == '100%'] = '99%'
                cond_vec = (ydata['stimFinger'] + ydata['cue'])[::n_channels].to_numpy()
                n_blocks = int(Y.shape[0] / 10)
                blocks = np.arange(n_blocks)
                part_vec = np.repeat(blocks, 10)
                Z = indicator(cond_vec)
                G = est_G_crossval(Y, Z, part_vec)[0]
                dist = G_to_dist(G)
                mask = ~np.eye(dist.shape[0], dtype=bool)
                D[p, tp] = dist
                Dav[p, tp] = dist[mask].mean()



    fig, axs = plt.subplots(1, n_timepoints, figsize=(12, 4.8))

    tick_labels = [
        'index,25%',
        'index,50%',
        'index,75%',
        'index,100%',
        'ring,0%',
        'ring,25%',
        'ring,50%',
        'ring,75%',
    ]
    vmax = D.mean(axis=0).max()
    vmin = D.mean(axis=0).min()
    for tp in range(n_timepoints):
        data2D = D.mean(axis=0)[tp]
        axs[tp].imshow(data2D, vmin=vmin, vmax=vmax)
        axs[tp].set_xticks(np.linspace(0, 7, len(tick_labels)))
        axs[tp].set_xticklabels(tick_labels, rotation=90)
        if tp == 0:
            axs[tp].set_yticks(np.linspace(0, 7, len(tick_labels)))
            axs[tp].set_yticklabels(tick_labels)

    plt.show()



