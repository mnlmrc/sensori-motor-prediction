import sys
import json
import nibabel as nb
import numpy as np
from joblib import Parallel, delayed

import pandas as pd
import rsatoolbox as rsa
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight

import globals as gl
import os

if __name__ == "__main__":
    participant_id = sys.argv[1]
    atlas = sys.argv[2]
    Hem = sys.argv[3]
    glm = sys.argv[4]

    experiment = 'smp1'

    pathGlm = os.path.join(gl.baseDir, experiment, gl.glmDir + glm, participant_id)
    pathSurf = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id)
    pathRDM = os.path.join(gl.baseDir, experiment, gl.RDM, participant_id)
    pathActivity = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id, 'beta')

    rois = ['BA1_exvivo',
            'BA2_exvivo',
            'BA3a_exvivo',
            'BA3b_exvivo',
            'BA4a_exvivo',
            'BA4p_exvivo',
            'BA6_exvivo']

    reginfo = pd.read_csv(f'{pathGlm}/{participant_id}_reginfo.tsv', sep='\t')
    cue = reginfo.cue.to_list()
    run = reginfo.run.to_list()

    A = nb.load(os.path.join(pathSurf, f'{participant_id}.{Hem}.32k.{atlas}.label.gii'))
    keys = A.darrays[0].data
    label = [(l.key, l.label) for l in A.labeltable.labels]
    label_df = pd.DataFrame(label, columns=['key', 'label'])

    B = nb.load(os.path.join(pathActivity, f'beta.{Hem}.func.gii'))

    rdm_eucl = list()
    rdm_maha = list()
    rdm_cv = list()
    for r, roi in enumerate(rois):
        print(f'processing {roi}...')

        key = label_df.key[label_df.label.to_list().index(roi)]
        data = np.array([b.data[keys == r] for b in B.darrays])

        dataset = rsa.data.Dataset(
            data,
            channel_descriptors={'channel': np.array(['vertex_' + str(x) for x in range(data.shape[-1])])},
            obs_descriptors={'cue': cue, 'run': run})

        # euclidean distance
        rdm_eucl.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='euclidean', descriptor='cue'))

        # mahalanobis
        noise = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_diag')
        rdm_maha.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='mahalanobis', descriptor='cue',
                                                    noise=noise))
        rdm_cv = rsa.rdm.calc_rdm_unbalanced(dataset, method='crossnobis', descriptor='cue',
                                             noise=noise, cv_descriptor='run')

    RDMs_eucl = np.concatenate([rdm.get_matrices() for rdm in rdm_eucl], axis=0)
    RDMs_maha = np.concatenate([rdm.get_matrices() for rdm in rdm_maha], axis=0)
    RDMs_cv = np.concatenate([rdm.get_matrices() for rdm in rdm_cv], axis=0)

    descr = json.dumps({
        'experiment': experiment,
        'participant': participant_id,
        'rdm_descriptors': dict(roi=tuple(rois)),
        'pattern_descriptors': {'cue': np.unique(cue)}
    })

    np.savez(os.path.join(pathRDM, f'RDMs.eucl.{atlas}.{Hem}.npz'),
             data_array=RDMs_eucl, descriptor=descr, allow_pickle=False
             )
    np.savez(os.path.join(pathRDM, f'RDMs.maha.{atlas}.{Hem}.npz'),
             data_array=RDMs_maha, descriptor=descr, allow_pickle=False
             )
    np.savez(os.path.join(pathRDM, f'RDMs.eucl.{atlas}.{Hem}.npz'),
             data_array=RDMs_cv, descriptor=descr, allow_pickle=False
             )

    # # visualize
    # rsa.vis.show_rdm(
    #     RDMs_eucl,
    #     show_colorbar='figure',
    #     rdm_descriptor='roi',
    #     pattern_descriptor='cue',
    #     n_row=1,
    #     figsize=(15, 5)
    # )
