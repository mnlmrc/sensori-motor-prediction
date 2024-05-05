import argparse
import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
import rsatoolbox as rsa

import globals as gl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument('--participant_id', default='subj100', help='Participant ID (e.g., subj100, subj101, ...)')

    args = parser.parse_args()

    participant_id = args.participant_id

    experiment = 'smp0'

    path = os.path.join(gl.baseDir, experiment, participant_id)
    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    npz = np.load(os.path.join(path, 'emg', f'smp0_{sn}_binned.npz'))
    emg = npz['data_array'][1:]
    emg = emg.reshape((emg.shape[1], emg.shape[2], emg.shape[0]))

    dat = pd.read_csv(os.path.join(path, f'smp0_{sn}.dat'), sep='\t')
    blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_emg.iloc[0].split('.')]
    dat = dat[dat.BN.isin(blocks)]
    channels = participants[participants['sn'] == sn].channels_emg.iloc[0].split(',')

    run = dat.BN.to_list()

    cue = dat.chordID.to_list()
    stimFinger = dat.stimFinger.to_list()
    time_point = np.linspace(0, 2, 3)
    timewin = ['SLR', 'LLR', 'Vol']

    map_cue = pd.DataFrame([('0%', 93),
                         ('25%', 12),
                         ('50%', 44),
                         ('75%', 21),
                         ('100%', 39)],
                        columns=['label', 'code'])
    map_dict = dict(zip(map_cue['code'], map_cue['label']))
    cue = [map_dict.get(item, item) for item in cue]

    index = [
        [1, 2, 3, 0],
        [0, 1, 2, 3]
    ]

    for sf, stimF in enumerate(np.unique(np.array(stimFinger))):

        dataset = rsa.data.TemporalDataset(
            emg[stimFinger == stimF],
            channel_descriptors={'channels': channels},
            obs_descriptors={'cue': [cue[i] for i in np.where(stimFinger == stimF)[0]],
                             'run': [run[i] for i in np.where(stimFinger == stimF)[0]]},
            time_descriptors={'time': time_point}
        )

        noise = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_diag')
        RDMs = rsa.rdm.calc_rdm_movie(dataset, method='mahalanobis', descriptor='cue',
                                                  noise=noise, cv_descriptor='run')

        RDMs.reorder(index[sf])

        RDMs.rdm_descriptors['timewin'] = [t for t in timewin]

        rsa.vis.show_rdm(RDMs,
                         pattern_descriptor='cue',
                         rdm_descriptor='timewin',
                         n_row=1,
                         figsize=(15, 3.5),
                         vmin=RDMs.get_matrices().min(), vmax=RDMs.get_matrices().max()
                         )

