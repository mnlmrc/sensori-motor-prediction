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
    parser.add_argument('--channel', default='index_flex', help='')
    parser.add_argument('--stimFinger', default='index', help='')

    args = parser.parse_args()

    experiment = args.experiment
    participant_ids = args.participants
    channel = args.channel
    stimFinger = args.stimFinger

    path = os.path.join(gl.baseDir, experiment)

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
    latency = latency[stimFinger].to_numpy()[0]

    map_cue = {
        '0%': 93,
        '25%': 12,
        '50%': 44,
        '75%': 21,
        '100%': 39
    }

    map_stimFinger = {
        'index': 91999,
        'ring': 99919
    }
    stimF = map_stimFinger[stimFinger]

    emg_all = list()

    for participant in participant_ids:

        sn = int(''.join([c for c in participant if c.isdigit()]))

        print(f'loading {participant}...')
        emg = np.load(os.path.join(path, participant, 'emg', f'{experiment}_{sn}.npy'))
        dat = pd.read_csv(os.path.join(path, participant, f'{experiment}_{sn}.dat'), sep='\t')
        blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_emg.iloc[0].split('.')]
        dat = dat[dat.BN.isin(blocks)]
        ch_p = participants[participants['sn'] == sn].channels_emg.iloc[0].split(',')

        # for ch in Dict.keys():
        #     if ch in ch_p:
        idx = ch_p.index(channel)
        emg_av = np.zeros((len(map_cue.items()), emg.shape[-1]))
        # for sf, stimF in enumerate(stimFinger_code):
        for c, cue in enumerate(map_cue.items()):
            emg_av[c] = emg[(dat.chordID == cue[1]) & (dat.stimFinger == stimF), idx].mean(axis=0)

        emg_all.append(emg_av)

    emg_all = np.array(emg_all)

    fig, axs = plt.subplots()

    tAx = np.linspace(-1, 2, emg_all.shape[-1]) - latency

    colors = make_colors(5)

    y = np.nanmean(emg_all, axis=0)
    for c, cue in enumerate(map_cue.items()):
        axs.plot(tAx, y[c], label=cue[0], color=colors[c])

    axs.set_title('Average EMG activity from one electrode placed over flexor digitorum '
                  'superficialis\nwhen the perturbation was delivered to the index finger')
    axs.set_ylabel('EMG (mV)')
    axs.set_xlabel('time relative to stimulation (s)')
    axs.set_xlim([-.05, .2])
    axs.legend()

    plt.show()





