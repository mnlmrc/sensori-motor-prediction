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
    parser.add_argument('--channels', default=['thumb_flex',
                                               'index_flex',
                                               'middle_flex',
                                               'ring_flex',
                                               'pinkie_flex',
                                               'thumb_ext',
                                               'index_ext',
                                               'middle_ext',
                                               'ring_ext',
                                               'pinkie_ext',
                                               'fdi'], help='')

    args = parser.parse_args()

    experiment = args.experiment
    participants = args.participants
    channels = args.channels

    path = os.path.join(gl.baseDir, experiment)

    df = pd.DataFrame()
    for participant in participants:
        # extract subject number
        sn = int(''.join([c for c in participant if c.isdigit()]))

        df_emg = pd.read_csv(os.path.join(path, participant, 'emg', f'{experiment}_{sn}_binned.tsv'), sep='\t')
        df_emg_grouped = df_emg.drop('participant_id', axis=1).groupby(['cue', 'stimFinger',
                                                                        'timewin']).mean().reset_index()
        df_emg_grouped['participant_id'] = participant

        # fill gaps in columns
        for ch in channels:
            if not ch in df_emg_grouped.columns:
                df_emg_grouped[ch] = np.nan

        df = pd.concat([df, df_emg_grouped])

    df = df.reset_index()
    colors = make_colors(5)
    palette = {cu: color for cu, color in zip(['0%', '25%', '50%', '75%', '100%'], colors)}

    fig, axs = plt.subplots(len(channels), len(df['stimFinger'].unique()),
                            sharey=True, sharex=True, figsize=(8, 10))
    for sf, stimF in enumerate(['index', 'ring']):
        subset = df[df['stimFinger'] == stimF]
        for c, ch in enumerate(channels):

            if (c == 0) & (sf == 0):
                axs[c, sf].set_title(f'stimFinger:Index\n{ch}')
            elif (c == 0) & (sf == 1):
                axs[c, sf].set_title(f'stimFinger:Ring\n{ch}')
            else:
                axs[c, sf].set_title(ch)



            sns.barplot(ax=axs[c, sf], data=subset, x='timewin', y=ch, hue='cue',
                          legend=False, palette=palette, hue_order=['0%', '25%', '50%', '75%', '100%'], log_scale=False)
            axs[c, sf].set_xlabel('')
            axs[c, sf].set_ylabel('')
            axs[c, sf].set_ylim([.5, 10])
            # axs[c, sf].set_yscale('log')

    # fig.legend(ncol=3, loc='upper left')
    fig.supylabel('EMG (% baseline)')
    # fig.suptitle(f'{participant_id}, emg')

    fig.tight_layout()

    # fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, 'force_bins.png'))

    plt.show()
