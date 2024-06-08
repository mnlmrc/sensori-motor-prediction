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
    parser.add_argument('--experiment', default='smp1', help='')
    parser.add_argument('--session', default='training', help='')
    parser.add_argument('--participants', default=['subj100',
                                                   'subj101',
                                                   'subj102',
                                                   'subj103',
                                                   'subj104',
                                                   'subj105',
                                                   'subj106',
                                                   # 'subj107',
                                                   # 'subj108',
                                                   # 'subj109',
                                                   # 'subj110'
                                                   ], help='')
    parser.add_argument('--channels', default=[
        'thumb',
        'index',
        'middle',
        'ring',
        'pinkie',

    ], help='')

    args = parser.parse_args()

    experiment = args.experiment
    participant_ids = args.participants
    channels = args.channels
    session = args.session

    path = os.path.join(gl.baseDir, experiment)

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    cue_code = [93, 12, 44, 21, 39]
    stimFinger_code = [91999, 99919]

    clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'mov', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
    latency = latency['index'][0], latency['ring'][0]

    Dict = {ch: [] for ch in channels}
    for participant in participant_ids:

        sn = int(''.join([c for c in participant if c.isdigit()]))

        print(f'loading {participant}...')
        if session == 'scanning':
            npz = np.load(os.path.join(path, gl.behavDir, participant, f'{experiment}_{sn}.npz'))
            force = npz['data_array']
            blocks = [int(b) for b in participants[participants['sn'] == sn].runsSess1.iloc[0].split('.')]
            dat = pd.read_csv(os.path.join(path, gl.behavDir, participant, f'{experiment}_{sn}.dat'), sep='\t')
            dat = dat[dat.stimFinger != 99999]
            dat = dat[dat.BN.isin(blocks)]
            stimFinger = dat.stimFinger
            cue = dat.cue
            ch_p = ['thumb', 'index', 'middle', 'ring', 'pinkie']
        elif session == 'training':
            npz = np.load(os.path.join(path, gl.trainDir, participant, f'{experiment}_{sn}.npz'))
            force = npz['data_array']
            blocks = [int(b) for b in participants[participants['sn'] == sn].runsSess1.iloc[0].split('.')]
            dat = pd.read_csv(os.path.join(path, gl.trainDir, participant, f'{experiment}_{sn}.dat'), sep='\t')
            dat = dat[dat.stimFinger != 99999]
            dat = dat[dat.BN.isin(blocks)]
            stimFinger = dat.stimFinger
            cue = dat.cue
            ch_p = ['thumb', 'index', 'middle', 'ring', 'pinkie']
        elif session == 'behavioural':
            force = np.load(os.path.join(path, participant, 'mov', f'{experiment}_{sn}.npy'))
            blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_mov.iloc[0].split('.')]
            dat = pd.read_csv(os.path.join(path, participant,f'{experiment}_{sn}.dat'), sep='\t')
            dat = dat[dat.BN.isin(blocks)]
            stimFinger = dat.stimFinger
            cue = dat.chordID
            ch_p = participants[participants['sn'] == sn].channels_mov.iloc[0].split(',')
        else:
            raise ValueError('Session name not recognized. Allowed session names are "scanning" and "training".')


        for ch in Dict.keys():
            if ch in ch_p:
                idx = ch_p.index(ch)
                force_av = np.zeros((len(cue_code), len(stimFinger_code), force.shape[-1]))
                for sf, stimF in enumerate(stimFinger_code):
                    for c, cue in enumerate(cue_code):
                        force_av[c, sf] = force[(dat.cue == cue) & (dat.stimFinger == stimF), idx].mean(axis=0)

                Dict[ch].append(force_av)

    colors = make_colors(5)
    palette = {cue: color for cue, color in zip(['0%', '25%', '50%', '75%', '100%'], colors)}

    tAx = np.linspace(-1, 2, force.shape[-1])
    tAx = tAx - latency[0], tAx - latency[1]

    # fig, axs = plt.subplots(len(channels), 2,
    #                         sharey=True, sharex=True, figsize=(6, 10))

    fig, axs = plt.subplots(1, 2,
                            sharey=True, sharex=True, figsize=(4, 6))

    if axs.ndim < 2:
        axs = np.expand_dims(axs, axis=0)

    for sf, stimF in enumerate(['index', 'ring']):
        for c, ch in enumerate(channels):

            if (c == 0) & (sf == 0):
                axs[0][sf].set_title(f'index perturbation')
            elif (c == 0) & (sf == 1):
                axs[0][sf].set_title(f'ring perturbation')
            else:
                pass # axs[c, sf].set_title(ch)



            y = np.nanmean(np.array(Dict[ch]), axis=0) + c * 8
            yerr = np.nanstd(np.array(Dict[ch]), axis=0) / np.sqrt(len(participants))

            # if (c < 5) & (sf==0):
            #     axs[0][sf].text(-.05, np.nanmean(y[:, 0, 0]), f'FDS$_{{{c}}}$', va='center', ha='right')
            # elif (c >= 5) & (c < 10) & (sf==0):
            #     axs[0][sf].text(-.05, np.nanmean(y[:, 0, 0]), f'EDS$_{{{c}}}$', va='center', ha='right')
            # elif  (c == 10) & (sf==0):
            #     axs[0][sf].text(-.05, np.nanmean(y[:, 0, 0]), 'FDI', va='center', ha='right')

            for col, color in enumerate(palette):
                axs[0][sf].plot(tAx[sf], y[col, sf], color=palette[color])
                axs[0][sf].fill_between(tAx[sf], y[col, sf] - yerr[col, sf], y[col, sf] + yerr[col, sf],
                                        color=palette[color], lw=0, alpha=.2)


            axs[0][sf].set_xlim([-.1, .5])
            # axs[0][sf].set_ylim([0, 1.15])
            axs[0][sf].spines[['top', 'bottom', 'right', ]].set_visible(False)

        axs[0][sf].axvline(0, ls='-', color='k', lw=.8)
        # axs[0][sf].axvline(.025, ls='--', color='k', lw=.8)
        # axs[0][sf].axvline(.05, ls='-.', color='k', lw=.8)
        # axs[0][sf].axvline(.1, ls=':', color='k', lw=.8)
        #
        # axs[0][sf].text(.025 + .0125, 1.1, 'SLR', ha='center')
        # axs[0][sf].text(.15, 1.1, 'Vol', ha='center')
        # axs[0][sf].text(.075, 1.1, 'LLR', ha='center')

    axs[0][0].spines[['bottom',]].set_visible(True)
    axs[0][1].spines[['bottom', ]].set_visible(True)

    axs[0][0].plot(tAx[0], clamp[0] + 1 * 8, color='k', ls='--')
    axs[0][1].plot(tAx[0], clamp[1] + 3 * 8, color='k', ls='--')

    labels = ['0%', '25%', '50%', '75%', '100%']
    for c, col in enumerate(colors):
        axs[0][0].plot(np.nan, label=labels[c], color=col)

    # fig.legend(ncol=5)

    fig.supxlabel('time relative to perturbation (s)')
    fig.supylabel('force (N)')

    fig.suptitle(f'{session} session')

    fig.tight_layout()

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', f'force.{session}.timec.svg'))

    plt.show()
