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

    RDMs_mat = np.zeros((len(participants), 2, 3, 4, 4))
    for p, participant in enumerate(participants):

        sn = int(''.join([c for c in participant if c.isdigit()]))

        npz = np.load(os.path.join(path, participant, 'emg', f'smp0_{sn}_RDMs.npz'))

        RDMs_mat[p] = npz['data_array']
        descr = json.loads(npz['descriptor'].item())

    RDMs_mat_av = RDMs_mat.mean(axis=0)

    stimFinger = ['index', 'ring']
    cue = {
        'index': ['25%', '50%', '75%', '100%'],
        'ring': ['0%', '25%', '50%', '75%']
    }
    timew = descr['rdm_descriptors']['timew']

    vmax = RDMs_mat_av.max()
    vmin = RDMs_mat_av.min()

    fig, axs = plt.subplots(2, 3, sharey='row')
    for sf, stimF in enumerate(stimFinger):
        for t, time in enumerate(timew):
            RDMs = rsa.rdm.RDMs(RDMs_mat_av[sf, t].reshape(1, 4, 4),
                                pattern_descriptors={'cue': cue[stimF]},
                                rdm_descriptors={'cond': f'{stimF}, {time}'})
            RDMs.n_rdm = 1
            RDMs.n_cond = 4

            if stimF is 'index':
                RDMs.reorder(np.array([1, 2, 3, 0]))

            rsa.vis.show_rdm_panel(RDMs,
                                   ax=axs[sf, t],
                                   vmin=vmin,
                                   vmax=vmax,
                                   rdm_descriptor='cond')

            axs[sf, t].set_xticks([0, 1, 2, 3])
            axs[sf, t].set_xticklabels(cue[stimF])
            axs[sf, t].set_yticks([0, 1, 2, 3])
            axs[sf, t].set_yticklabels(cue[stimF])

