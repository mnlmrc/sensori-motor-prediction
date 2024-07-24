import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import globals as gl
from visual import make_colors


def plot_timec(data, channels=None, xlim=None, ylim=None, title=None, clamp=None, vsep=8):
    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
    latency = latency['index'][0], latency['ring'][0]

    tAx = (np.linspace(-gl.prestim, gl.poststim, data.shape[-1]) - latency[0],
           np.linspace(-gl.prestim, gl.poststim, data.shape[-1]) - latency[1])

    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(4, 6))

    colors = make_colors(5)
    palette = {cue: color for cue, color in zip(gl.clabels, colors)}

    for sf, stimF in enumerate(['index', 'ring']):
        ax = axs[sf]
        ax.set_title(f'{stimF} perturbation')

        for c, ch in enumerate(channels):
            y = data.mean(axis=0)[:, sf, c] + c * vsep
            yerr = data.std(axis=0)[:, sf, c] / np.sqrt(data.shape[0])

            for col, color in enumerate(palette):
                ax.plot(tAx[sf], y[col], color=palette[color])
                # ax.fill_between(tAx[sf], y - yerr, y + yerr,
                #                 color=palette[color], lw=0, alpha=.2)

            ax.set_xlim(xlim)
            ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)

        ax.axvline(0, ls='-', color='k', lw=.8)
        ax.set_yticks([])  # Remove y-ticks

    axs[0].set_ylim(ylim)

    for ax in axs:
        ax.spines[['bottom']].set_visible(True)

    if clamp is not None:
        for i, ax in enumerate(axs):
            ax.plot(tAx[i], clamp[i] + (1 + 2 * i) * vsep, color='k', ls='--')
        axs[0].plot(np.nan, color='k', ls='--', label='clamped')

        # Add a vertical line for y-scale reference
        reference_length = 5  # Length of the reference line
        reference_x = xlim[0]  # Position of the reference line
        midpoint_y = (ylim[0] + ylim[1]) / 2  # Calculate the midpoint of the y-axis
        axs[0].plot([reference_x, reference_x], [midpoint_y - reference_length / 2, midpoint_y + reference_length / 2],
                    ls='-', color='k', lw=3)
        axs[0].text(reference_x, midpoint_y, f'{reference_length}N ', color='k',
                    ha='right', va='center')

    for c, col in enumerate(colors):
        axs[0].plot(np.nan, label=gl.clabels[c], color=col)

    fig.legend(ncol=3, loc='upper center')
    fig.supxlabel('time relative to perturbation (s)')
    fig.suptitle(title, y=.9)
    fig.subplots_adjust(top=.82)

    return fig, axs


if __name__ == "__main__":
    clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
    latency = latency['index'][0], latency['ring'][0]

    plot_timec('smp0',
               'behav',
               channels=gl.channels['mov'],
               clamp=clamp,
               xlim=[-.1, .5],
               ylim=[0, 40],
               filename='force.timec.behav.png')

    plot_timec('smp1',
               'training',
               channels=gl.channels['mov'],
               clamp=clamp,
               xlim=[-.1, .5],
               ylim=[0, 40],
               filename='force.timec.training.png')

    plot_timec('smp1',
               'scanning',
               channels=gl.channels['mov'],
               clamp=clamp,
               xlim=[-.1, .5],
               ylim=[0, 40],
               filename='force.timec.scanning.png')

    plot_timec('smp2',
               'pilot',
               channels=gl.channels['mov'],
               clamp=clamp,
               xlim=[-.1, .5],
               ylim=[0, 40],
               filename='force.timec.pilot.png')

    plt.show()
