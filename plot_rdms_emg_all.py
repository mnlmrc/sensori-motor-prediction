import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import globals as gl
import rsatoolbox as rsa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
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
                                                   'subj110'], help='Participant IDs')

    args = parser.parse_args()

    participants = args.participants

    experiment = 'smp0'

    path = os.path.join(gl.baseDir, experiment)

    RDMs_mat = np.zeros((len(participants), 3, 8, 8))
    for p, participant in enumerate(participants):
        sn = int(''.join([c for c in participant if c.isdigit()]))

        npz = np.load(os.path.join(path, participant, 'emg', f'smp0_{sn}_RDMs.npz'))

        RDMs_mat[p] = npz['data_array']
        descr = json.loads(npz['descriptor'].item())

    RDMs_mat_av = RDMs_mat.mean(axis=0)

    timew = descr['rdm_descriptors']['timew']

    vmax = RDMs_mat_av.max()
    vmin = RDMs_mat_av.min()

    fig, axs = plt.subplots(1, 3, sharey='row')
    # for sf, stimF in enumerate(stimFinger):
    for t, time in enumerate(timew):
        RDMs = rsa.rdm.RDMs(RDMs_mat_av[t].reshape(1, 8, 8),
                            pattern_descriptors=descr['pattern_descriptors'],
                            rdm_descriptors={'cond': f'{time}'})

        rsa.vis.show_rdm_panel(RDMs,
                               ax=axs[t],
                               vmin=vmin,
                               vmax=vmax,
                               rdm_descriptor='cond',
                               cmap='viridis')

        axs[ t].axvline(3.5, color='k', lw=.8)
        axs[t].axhline(3.5, color='k', lw=.8)

        axs[t].set_xticks(np.linspace(0, 7, 8))
        axs[t].set_xticklabels(RDMs.pattern_descriptors['stimFinger,cue'], rotation=45, ha='right')
        axs[t].set_yticks(np.linspace(0, 7, 8))
        axs[t].set_yticklabels(RDMs.pattern_descriptors['stimFinger,cue'])

        # rsa.vis.scatter_plot.show_MDS_panel(RDMs,
        #                                     axs[1, t],
        #                                     pattern_descriptor='stimFinger,cue')
        # axs[1, t].set_xlim([-3, 3])
        # axs[1, t].set_ylim([-3, 3])

    fig.suptitle('RDMs, emg')

