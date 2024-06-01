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
    parser.add_argument('--participant_id', default='subj103', help='Participant ID')
    parser.add_argument('--experiment', default='smp0', help='')

    args = parser.parse_args()

    participant_id = args.participant_id
    experiment = args.experiment

    # extract subject number
    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    path = os.path.join(gl.baseDir, experiment, participant_id)

    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')

    emg = np.load(os.path.join(path, 'emg', f'{experiment}_{sn}.npy'))
    dat = pd.read_csv(os.path.join(path, f'{experiment}_{sn}.dat'), sep='\t')

    channels = participants[participants['sn'] == sn].channels_emg.iloc[0].split(',')
    blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_emg.iloc[0].split('.')]
    dat = dat[dat.BN.isin(blocks)]

    stimFinger = dat['stimFinger']
    cue = dat['chordID']

    # map codes to actual labels
    map_cue = pd.DataFrame([('0%', 93),
                            ('25%', 12),
                            ('50%', 44),
                            ('75%', 21),
                            ('100%', 39)],
                           columns=['cue', 'code'])

    map_stimFinger = pd.DataFrame([('index', 91999),
                                   ('ring', 99919)],
                                  columns=['stimFinger', 'code'])

    map_cue_dict = dict(zip(map_cue['code'], map_cue['cue']))
    map_stimFinger_dict = dict(zip(map_stimFinger['code'], map_stimFinger['stimFinger']))

    cue = [map_cue_dict.get(item, item) for item in cue]
    stimFinger = [map_stimFinger_dict.get(item, item) for item in stimFinger]

    # define time windows
    fsample = 2148
    prestim = int(1 * fsample)
    poststim = int(2 * fsample)
    win = {'Pre': (prestim - int(.1 * fsample), prestim),
           'SLR': (prestim + int(.025 * fsample), prestim + int(.05 * fsample)),
           'LLR': (prestim + int(.05 * fsample), prestim + int(.1 * fsample)),
           'Vol': (prestim + int(.1 * fsample), prestim + int(.5 * fsample))}

    # compute averages in time windows
    emg_binned = np.zeros((len(win.keys()), emg.shape[0], emg.shape[1]))
    for k, key in enumerate(win.keys()):
        emg_binned[k, dat.stimFinger == 91999] = emg[dat.stimFinger == 91999, :,
                                               win[key][0] + int(latency['index'].iloc[0] * fsample):
                                               win[key][1] + int(latency['index'].iloc[0] * fsample)
                                               ].mean(axis=-1)
        emg_binned[k, dat.stimFinger == 99919] = emg[dat.stimFinger == 99919, :,
                                               win[key][0] + int(latency['ring'].iloc[0] * fsample):
                                               win[key][1] + int(latency['ring'].iloc[0] * fsample)
                                               ].mean(axis=-1)

    emg_binned /= emg_binned[0]

    df_emg = pd.DataFrame(data=emg_binned.reshape((-1, emg.shape[1])), columns=channels)
    df_emg['stimFinger'] = stimFinger * len(win.keys())
    df_emg['cue'] = cue * len(win.keys())
    df_emg['timewin'] = np.concatenate([[key] * emg.shape[0] for key in win.keys()])
    df_emg['participant_id'] = participant_id

    descr = {
        'fsample': fsample,
        'prestim': prestim,
        'poststim': poststim,
        'time windows': win
    }

    df_emg.to_csv(os.path.join(path, 'emg', f'smp0_{sn}_binned.tsv'), sep='\t')
    np.savez(os.path.join(path, 'emg', f'{experiment}_{sn}_binned.npz'),
             data_array=emg_binned, descriptor=descr, allow_pickle=False)

    colors = make_colors(5)
    palette = {cu: color for cu, color in zip(map_cue['cue'], colors)}

    fig, axs = plt.subplots(len(channels), len(np.unique(stimFinger)),
                            sharey=True, sharex=True, figsize=(8, 10))
    for c, ch in enumerate(channels):
        for sf, stimF in enumerate(np.unique(stimFinger)):

            if (c == 0) & (sf == 0):
                axs[c, sf].set_title(f'stimFinger:Index\n{ch}')
            elif (c == 0) & (sf == 1):
                axs[c, sf].set_title(f'stimFinger:Ring\n{ch}')
            else:
                axs[c, sf].set_title(ch)

            subset = df_emg[df_emg['stimFinger'] == stimF]

            sns.boxplot(ax=axs[c, sf], data=subset, x='timewin', y=ch, hue='cue',
                        legend=False, palette=palette, hue_order=['0%', '25%', '50%', '75%', '100%'])
            axs[c, sf].set_xlabel('')
            axs[c, sf].set_ylabel('')
            axs[c, sf].set_ylim([0, 30])
            axs[c, sf].set_yscale('linear')

    # fig.legend(ncol=3, loc='upper left')
    fig.supylabel('EMG (% baseline)')
    fig.suptitle(f'{participant_id}, emg')

    fig.tight_layout()

    # fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, 'force_bins.png'))

    plt.show()

