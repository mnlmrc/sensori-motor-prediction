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
    participant_ids = args.participants
    channels = args.channels

    path = os.path.join(gl.baseDir, experiment)

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    cue_code = [93, 12, 44, 21, 39]
    stimFinger_code = [91999, 99919]

    Dict = {ch: [] for ch in channels}
    for participant in participant_ids:

        sn = int(''.join([c for c in participant if c.isdigit()]))

        print(f'loading {participant}...')
        emg = np.load(os.path.join(path, participant, 'emg', f'{experiment}_{sn}.npy'))
        dat = pd.read_csv(os.path.join(path, participant, f'{experiment}_{sn}.dat'), sep='\t')
        blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_emg.iloc[0].split('.')]
        dat = dat[dat.BN.isin(blocks)]
        ch_p = participants[participants['sn'] == sn].channels_emg.iloc[0].split(',')

        for ch in Dict.keys():
            if ch in ch_p:
                idx = ch_p.index(ch)
                emg_av = np.zeros((len(cue_code), len(stimFinger_code), emg.shape[-1]))
                for sf, stimF in enumerate(stimFinger_code):
                    for c, cue in enumerate(cue_code):
                        emg_av[c, sf] = emg[(dat.chordID == cue) & (dat.stimFinger == stimF), idx].mean(axis=0)

                Dict[ch].append(emg_av)

    colors = make_colors(5)
    palette = {cue: color for cue, color in zip(['0%', '25%', '50%', '75%', '100%'], colors)}

    tAx = np.linspace(-1, 2, emg.shape[-1])

    fig, axs = plt.subplots(len(channels), 2,
                            sharey=True, sharex=True, figsize=(8, 10))
    for sf, stimF in enumerate(['index', 'ring']):
        for c, ch in enumerate(channels):

            if (c == 0) & (sf == 0):
                axs[c, sf].set_title(f'stimFinger:Index\n{ch}')
            elif (c == 0) & (sf == 1):
                axs[c, sf].set_title(f'stimFinger:Ring\n{ch}')
            else:
                axs[c, sf].set_title(ch)

            subset = np.nanmean(np.array(Dict[ch]), axis=0)

            for col, color in enumerate(palette):
                axs[c, sf].plot(tAx, subset[col, sf], color=palette[color])

            axs[c, sf].set_xlim([-.1, .5])


    fig.tight_layout()

    # fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, 'force_bins.png'))

    plt.show()
