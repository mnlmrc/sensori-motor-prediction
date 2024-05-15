import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import globals as gl
import rsatoolbox as rsa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--participant_id', default='subj100', help='Participant ID')

    args = parser.parse_args()

    participant_id = args.participant_id

    experiment = 'smp0'

    path = os.path.join(gl.baseDir, experiment)

    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    channels = participants[participants['sn'] == sn].channels_mov.iloc[0].split(',')
    blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_mov.iloc[0].split('.')]
    dat = pd.read_csv(os.path.join(path, participant_id, f'{experiment}_{sn}.dat'), sep='\t')
    dat = dat[dat.BN.isin(blocks)]

    cue = dat.chordID
    stimFinger = dat.stimFinger
    run = dat.BN

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

    # load RDM(s)
    npz = np.load(os.path.join(path, participant_id, 'mov', f'smp0_{sn}_RDMs.npz'))
    RDMs = npz['data_array']
    descr = json.loads(npz['descriptor'].item())

    # load force
    force = np.load(os.path.join(path, participant_id, 'mov', f'smp0_{sn}.npy'))

    timeAx = np.linspace(-1, 2, force.shape[-1])
    dist_stimFinger = np.zeros(force.shape[-1])
    dist_cue = np.zeros(force.shape[-1])
    for t in range(force.shape[-1]):

        print('time point %f' % timeAx[t])

        # calculate rdm for timepoint t
        force_tmp = force[:, :, t]
        dataset = rsa.data.Dataset(
            force_tmp,
            channel_descriptors={'channels': channels},
            obs_descriptors={'stimFinger,cue': [sf + ',' + c for c, sf in zip(cue, stimFinger)], 'run': run},
        )
        noise = rsa.data.noise.prec_from_unbalanced(dataset,
                                                    obs_desc='stimFinger,cue',
                                                    method='shrinkage_diag')
        rdm = rsa.rdm.calc_rdm_unbalanced(dataset,
                                          method='crossnobis',
                                          descriptor='stimFinger,cue',
                                          noise=noise,
                                          cv_descriptor='run')
        rdm.reorder(rdm.pattern_descriptors['stimFinger,cue'].argsort())
        rdm.reorder(np.array([1, 2, 3, 0, 4, 5, 6, 7]))

        rdm = rdm.get_matrices()

        dist_stimFinger[t] = rdm[0, [0, 1, 2], [5, 6, 7]].mean()
        dist_cue[t] = rdm[0, [0, 1, 1], [1, 2, 2]].mean()

    fig, axs = plt.subplots()

    axs.plot(timeAx, dist_stimFinger)
    axs.plot(timeAx, dist_cue)