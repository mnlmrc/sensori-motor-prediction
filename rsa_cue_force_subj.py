import argparse
import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
import rsatoolbox as rsa

import matplotlib.pyplot as plt

import globals as gl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument('--participant_id', default='subj100', help='Participant ID (e.g., subj100, subj101, ...)')
    parser.add_argument('--channels', nargs='+', default=['thumb', 'index', 'middle', 'ring', 'pinkie'], help='')
    parser.add_argument('--method', default='euclidean', help='')

    args = parser.parse_args()

    participant_id = args.participant_id
    channels = args.channels
    method = args.method

    experiment = 'smp0'

    path = os.path.join(gl.baseDir, experiment, participant_id)
    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    npz = np.load(os.path.join(path, 'mov', f'smp0_{sn}_binned.npz'))
    force = npz['data_array']
    # force = force.reshape((force.shape[1], force.shape[2], force.shape[0]))

    dat = pd.read_csv(os.path.join(path, f'smp0_{sn}.dat'), sep='\t')
    blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_mov.iloc[0].split('.')]
    dat = dat[dat.BN.isin(blocks)]
    channels_mov = participants[participants['sn'] == sn].channels_mov.iloc[0].split(',')
    # force = force[:, :, np.array([channels_mov.index(ch) for ch in channels])]

    run = dat.BN.to_list()

    cue = dat.chordID.to_list()
    stimFinger = dat.stimFinger.to_list()
    time_point = np.linspace(0, 2, 3)
    timewin = ['Pre', 'LLR', 'Vol']

    map_cue = pd.DataFrame([('0%', 93),
                            ('25%', 12),
                            ('50%', 44),
                            ('75%', 21),
                            ('100%', 39)],
                           columns=['label', 'code'])
    map_dict = dict(zip(map_cue['code'], map_cue['label']))
    cue = [map_dict.get(item, item) for item in cue]

    map_stimFinger = pd.DataFrame([('index', 91999),
                                   ('ring', 99919), ],
                                  columns=['label', 'code'])
    map_dict = dict(zip(map_stimFinger['code'], map_stimFinger['label']))
    stimFinger = [map_dict.get(item, item) for item in stimFinger]

    # dataset = rsa.data.TemporalDataset(
    #     force,
    #     channel_descriptors={'channels': channels},
    #     obs_descriptors={'stimFinger,cue': [sf + ',' + c for c, sf in zip(cue, stimFinger)], 'run': run},
    #     time_descriptors={'time': time_point}
    # )
    #
    # dataset_split = dataset.split_time('time')

    rdms = list()
    for f in range(force.shape[0]):
        dataset = rsa.data.Dataset(
            force[f],
            channel_descriptors={'channels': channels},
            obs_descriptors={'stimFinger,cue': [sf + ',' + c for c, sf in zip(cue, stimFinger)], 'run': run},
        )
        noise = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='stimFinger,cue', method='shrinkage_diag')
        rdms.append(rsa.rdm.calc_rdm_unbalanced(dataset, method=method, descriptor='stimFinger,cue',
                                                noise=noise, cv_descriptor='run'))
        # rdms.append(rsa.rdm.calc_rdm_unbalanced(ds, method='euclidean', descriptor='stimFinger,cue'))

    rdms = rsa.rdm.concat(rdms)
    rdms.rdm_descriptors = {'timewin': timewin}

    # put labels in alphabetical order
    index = rdms.pattern_descriptors['stimFinger,cue'].argsort()
    rdms.reorder(index)

    # adjust order
    index = [1, 2, 3, 0, 4, 5, 6, 7]
    rdms.reorder(index)

    fig, axs = plt.subplots(2, len(timewin), figsize=(15, 6) )
    for r, rdm in enumerate(rdms):
        cax = rsa.vis.show_rdm_panel(rdm,
                                     ax=axs[0, r],
                                     rdm_descriptor='timewin',
                                     vmin=rdms.get_matrices().min(),
                                     vmax=rdms.get_matrices().max())

        axs[0, r].axvline(3.5, color='k', lw=.8)
        axs[0, r].axhline(3.5, color='k', lw=.8)

        axs[0, r].set_xticks(np.linspace(0, 7, 8))
        axs[0, r].set_xticklabels(rdm.pattern_descriptors['stimFinger,cue'], rotation=45, ha='right')
        axs[0, r].set_yticks(np.linspace(0, 7, 8))
        axs[0, r].set_yticklabels(rdm.pattern_descriptors['stimFinger,cue'])

        rsa.vis.scatter_plot.show_MDS_panel(rdm,
                                            axs[1, r],
                                            pattern_descriptor='stimFinger,cue')
        # axs[1, r].set_xlim([-3, 3])
        # axs[1, r].set_ylim([-3, 3])

    # Create a colorbar
    cbar = fig.colorbar(cax, ax=axs, orientation='horizontal', fraction=.02)
    cbar.set_label('cross-validated multivariate distance (a.u.)')

    fig.subplots_adjust(bottom=.4)

    # RDMs.append(rdm)

    descr = json.dumps({
        'experiment': experiment,
        'participant': participant_id,
        'pattern_descriptors': rdms.pattern_descriptors,
        'rdm_descriptors': {'timew': ['Pre', 'LLR', 'Vol']}
    })

    # RDMs = np.array([RDMs[0].get_matrices(), RDMs[1].get_matrices()])
    RDMs = rdms.get_matrices()

    np.savez(os.path.join(path, 'mov', f'smp0_{sn}_RDMs.npz'),
             data_array=RDMs, descriptor=descr, allow_pickle=False)

