import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import globals as gl
from experiment import Param
from visual import make_colors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--experiment', default='smp0', help='')
    parser.add_argument('--participants', default=['subj100',
                                                   'subj101',
                                                   'subj102',
                                                   'subj103',
                                                   'subj104',
                                                   'subj105',
                                                   'subj106',
                                                   'subj107',
                                                   'subj108',
                                                   'subj109',
                                                   'subj110'], help='')
    parser.add_argument('--channels', default=['thumb',
                                               'index',
                                               'middle',
                                               'ring',
                                               'pinkie'
                                               ], help='')

    args = parser.parse_args()

    experiment = args.experiment
    participants = args.participants
    channels = args.channels

    path = os.path.join(gl.baseDir, experiment)

    df = pd.DataFrame()
    for participant in participants:
        # extract subject number
        sn = int(''.join([c for c in participant if c.isdigit()]))

        df_force = pd.read_csv(os.path.join(path, participant, 'mov', f'{experiment}_{sn}_binned.tsv'), sep='\t')
        df_force_grouped = df_force.drop('participant_id', axis=1).groupby(['cue', 'stimFinger',
                                                                            'timewin']).mean().reset_index()
        df_force_grouped['participant_id'] = participant

        # fill gaps in columns
        for ch in channels:
            if not ch in df_force_grouped.columns:
                df_force_grouped[ch] = np.nan

        df = pd.concat([df, df_force_grouped])

    df = df.reset_index()
    colors = make_colors(5)
    palette = {cu: color for cu, color in zip(['0%', '25%', '50%', '75%', '100%'], colors)}

    fig, axs = plt.subplots(len(channels), len(df['stimFinger'].unique()),
                            sharey=True, sharex=True, figsize=(8, 10))
    for sf, stimF in enumerate(['index', 'ring']):
        subset = df[(df['stimFinger'] == stimF)]
        for c, ch in enumerate(channels):

            if (c == 0) & (sf == 0):
                axs[c, sf].set_title(f'stimFinger:Index\n{ch}')
            elif (c == 0) & (sf == 1):
                axs[c, sf].set_title(f'stimFinger:Ring\n{ch}')
            else:
                axs[c, sf].set_title(ch)

            sns.boxplot(ax=axs[c, sf], data=subset, x='timewin', y=ch, hue='cue', order=['Pre', 'LLR', 'Vol'],
                        palette=palette, hue_order=['0%', '25%', '50%', '75%', '100%'], legend=False)
            axs[c, sf].set_xlabel('')
            axs[c, sf].set_ylabel('')
            # axs[c, sf].set_ylim([.5, 15])
            # axs[c, sf].set_yscale('log')

    fig.legend(ncol=5, loc='upper left')
    fig.supylabel('Force (N)')
    # fig.suptitle(f'{participant_id}, emg')

    fig.tight_layout()

    fig, axs = plt.subplots()

    sns.scatterplot(ax=axs, data=df[df['timewin'] == 'LLR'], x='index', y='ring', style='stimFinger', hue='cue', palette=palette)

    # Calculate the center of mass for each group
    centers = df[df['timewin'] == 'LLR'].groupby(['stimFinger', 'cue']).agg({'index': 'mean', 'ring': 'mean'}).reset_index()

    # Plot the centers of mass
    # You can change the marker and size here to differentiate from other points
    color = list(palette.items())
    color = [c[1] for c in color]
    color = [color[4], color[1], color[2], color[3], color[0], color[1], color[2], color[3]]
    for r, row in centers.iterrows():
        if r < 4:
            axs.scatter(row['index'], row['ring'], color=color[r], marker='x', s=100,
                        label='Center of Mass' if r == 0 else "")
        else:
            axs.scatter(row['index'], row['ring'], color=color[r], marker='o', s=100,
                        label='Center of Mass' if r == 0 else "")


    # # Optional: Add legend for clarity if needed
    # handles, labels = axs.get_legend_handles_labels()
    # # Create a handle for the center of mass points and add it to the legend
    # from matplotlib.lines import Line2D
    #
    # legend_line = Line2D([0], [0], marker='x', color='w', label='Center of Mass', markerfacecolor='black',
    #                      markersize=10)
    # handles.append(legend_line)
    # axs.legend(handles=handles)
    # fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, 'force_bins.png'))

    plt.show()
