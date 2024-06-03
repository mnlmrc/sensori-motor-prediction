import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import globals as gl
from experiment import Param
from visual import make_colors

if __name__ == "__main__":
    experiment = 'smp1'
    participant_id = 'subj102'
    session = 'scanning'

    # extract subject number
    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    if session == 'scanning':
        path = os.path.join(gl.baseDir, experiment, gl.behavDir, participant_id)
    elif session == 'training':
        path = os.path.join(gl.baseDir, experiment, gl.trainDir, participant_id)
    else:
        raise ValueError('Session name not recognized. Allowed session names are "scanning" and "training".')

    npz = np.load(os.path.join(path, f'{experiment}_{sn}.npz'))
    clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'mov', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

    force = npz['data_array']
    descr = json.loads(npz['descriptor'].item())

    stimFinger = pd.Series(descr['trial_info']['stimFinger'])
    cue = pd.Series(descr['trial_info']['cue'])
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinkie']
    columns = pd.Series(descr['columns'])

    cues = pd.DataFrame([('0%', '93'),
                         ('25%', '12'),
                         ('50%', '44'),
                         ('75%', '21'),
                         ('100%', '39')],
                        columns=['label', 'instr'])

    timeAx = Param().timeAx()

    colors = make_colors(5)

    fig, axs = plt.subplots(len(fingers), len(np.unique(stimFinger)),
                            sharey=True, sharex=True, figsize=(8, 10))
    for f, fi in enumerate(fingers):
        for sf, stimF in enumerate(np.unique(stimFinger)):

            # plot clamped
            axs[f, sf].plot(timeAx, clamp[sf], color='k', ls='--', lw=.8, label='clamped')

            # add vlines
            axs[f, sf].axvline(0, lw=.8, color='k', ls='-')
            axs[f, sf].axvline(.2, lw=.8, color='k', ls='-.')
            axs[f, sf].axvline(.5, lw=.8, color='k', ls=':')
            for c, cu in enumerate(cues.label.to_list()):

                # plot force
                y = force[(stimFinger == stimF) & (cue == cues.instr.iloc[c]), f, :].mean(axis=0)
                yerr = force[(stimFinger == stimF) & (cue == cues.instr.iloc[c]), f, :].std(axis=0)
                axs[f, sf].plot(timeAx, y, color=colors[c], label=cu)
                axs[f, sf].fill_between(timeAx, y - yerr, y + yerr, color=colors[c], alpha=.2, lw=0)

                if (f == 0) & (sf == 0):
                    axs[f, sf].set_title(f'stimFinger:Index\n{fi}')
                elif (f == 0) & (sf == 1):
                    axs[f, sf].set_title(f'stimFinger:Ring\n{fi}')
                else:
                    axs[f, sf].set_title(fi)

    axs[0, 0].set_xlim([-.1, 1])
    fig.supxlabel('time relative to stimulation (s)')
    fig.supylabel('force (N)')
    fig.suptitle(f'{participant_id}, session: {session}')

    fig.tight_layout()

    axs[0, 1].legend(ncol=6, loc='upper right')

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, f'force_timec_{session}.png'))

    plt.show()
