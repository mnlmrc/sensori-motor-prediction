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
    parser.add_argument('--participant_id', default='subj110', help='Participant ID')
    parser.add_argument('--experiment', default='smp0', help='')
    parser.add_argument('--session', default='behav', help='')

    args = parser.parse_args()

    participant_id = args.participant_id
    experiment = args.experiment
    session = args.session

    # extract subject number
    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    if session == 'scanning':
        path = os.path.join(gl.baseDir, experiment, gl.behavDir, participant_id)
    elif session == 'training':
        path = os.path.join(gl.baseDir, experiment, gl.trainDir, participant_id)
    elif session == 'behav':
        path = os.path.join(gl.baseDir, experiment, participant_id)
    else:
        raise ValueError('Session name not recognized. Allowed session names are "scanning" and "training".')

    force = np.load(os.path.join(path, 'mov', f'{experiment}_{sn}.npy'))
    dat = pd.read_csv(os.path.join(path, f'{experiment}_{sn}.dat'), sep='\t')

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_mov.iloc[0].split('.')]
    dat = dat[dat.BN.isin(blocks)]

    stimFinger = dat.stimFinger
    cue = dat.chordID
    channels = ['thumb', 'index', 'middle', 'ring', 'pinkie']
    # columns = pd.Series(descr['columns'])

    map_cue = pd.DataFrame([('0%', 93),
                         ('25%', 12),
                         ('50%', 44),
                         ('75%', 21),
                         ('100%', 39)],
                        columns=['label', 'instr'])
    map_dict = dict(zip(map_cue['instr'], map_cue['label']))
    cue = [map_dict.get(item, item) for item in cue]

    map_stimFinger = pd.DataFrame([('index', 91999),
                                   ('ring', 99919), ],
                                  columns=['label', 'code'])
    map_dict = dict(zip(map_stimFinger['code'], map_stimFinger['label']))
    stimFinger = [map_dict.get(item, item) for item in stimFinger]

    # define time windows
    fsample = 500
    prestim = int(1 * fsample)
    poststim = int(2 * fsample)
    win = {'Pre': (prestim - int(.5 * fsample), prestim),
           'LLR': (prestim + int(.2 * fsample), prestim + int(.5 * fsample)),
           'Vol': (prestim + int(.5 * fsample), prestim + int(1 * fsample))}

    # compute averages in time windows
    force_binned = np.zeros((len(win.keys()), force.shape[0], force.shape[1]))
    for k, key in enumerate(win.keys()):
        force_binned[k, dat.stimFinger == 91999] = force[dat.stimFinger == 91999, :,
                                                   win[key][0]:
                                                   win[key][1]
                                                   ].mean(axis=-1)
        force_binned[k, dat.stimFinger == 99919] = force[dat.stimFinger == 99919, :,
                                                   win[key][0]:
                                                   win[key][1]
                                                   ].mean(axis=-1)

    # force_binned /= force_binned[0]

    df_force = pd.DataFrame(data=force_binned.reshape((-1, force.shape[1])), columns=channels)
    df_force['stimFinger'] = stimFinger * len(win.keys())
    df_force['cue'] = cue * len(win.keys())
    df_force['timewin'] = np.concatenate([[key] * force.shape[0] for key in win.keys()])
    df_force['participant_id'] = participant_id

    descr = {
        'fsample': fsample,
        'prestim': prestim,
        'poststim': poststim,
        'time windows': win
    }

    df_force.to_csv(os.path.join(path, 'mov', f'smp0_{sn}_binned.tsv'), sep='\t')
    np.savez(os.path.join(path, 'mov', f'{experiment}_{sn}_binned.npz'),
             data_array=force_binned, descriptor=descr, allow_pickle=False)

    colors = make_colors(5)
    palette = {cu: color for cu, color in zip(map_cue['label'], colors)}

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

            subset = df_force[df_force['stimFinger'] == stimF]

            sns.boxplot(ax=axs[c, sf], data=subset, x='timewin', y=ch, hue='cue',
                        legend=False, palette=palette, hue_order=['0%', '25%', '50%', '75%', '100%'])
            axs[c, sf].set_xlabel('')
            axs[c, sf].set_ylabel('')
            axs[c, sf].set_ylim([0, 30])
            axs[c, sf].set_yscale('linear')

    # fig.legend(ncol=3, loc='upper left')
    fig.supylabel('Force (N)')
    fig.suptitle(f'{participant_id}, force')

    fig.tight_layout()

    # fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, 'force_bins.png'))

    plt.show()
