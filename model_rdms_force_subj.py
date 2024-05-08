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

        npz = np.load(os.path.join(path, participant, 'mov', f'smp0_{sn}_RDMs.npz'))

        RDMs_mat[p] = npz['data_array']
        descr = json.loads(npz['descriptor'].item())

    RDMs_mat_av = RDMs_mat.mean(axis=0)

    # create finger model
    stimFinger_RDM = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if (i < 4 and j >= 4) or (i >= 4 and j < 4):
                stimFinger_RDM[i, j] = 1  # Different fingers

    # create cue model
    cue_RDM = np.zeros((8, 8))
    max_cue_diff = 3
    for i in range(8):
        for j in range(8):
            cue_diff = abs((i % 4) - (j % 4))
            cue_RDM[i, j] = cue_diff / max_cue_diff

    timew = descr['rdm_descriptors']['timew']

    vmax = RDMs_mat_av.max()
    vmin = RDMs_mat_av.min()

    rdms = rsa.rdm.RDMs(np.stack([stimFinger_RDM, cue_RDM, np.zeros((8, 8))]))
    model = rsa.model.ModelWeighted('test', rdms)

    for t, time in enumerate(timew):
        RDMs = rsa.rdm.RDMs(RDMs_mat_av[t].reshape(1, 8, 8),
                            pattern_descriptors=descr['pattern_descriptors'],
                            rdm_descriptors={'cond': f'{time}'})

        theta = model.fit(RDMs)
        pred = model.predict(theta)

        obs = RDMs.dissimilarities.squeeze()

        res_squared = ((obs - pred)**2).mean()
        print(res_squared)




