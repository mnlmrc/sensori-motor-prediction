import nibabel as nb
import numpy as np
from joblib import Parallel, delayed

import pandas as pd
import rsatoolbox as rsa
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight

import globals as gl
import os

participant_id = 'subj100'
experiment = 'smp1'
glm = '1'

pathGlm = os.path.join(gl.baseDir, experiment, gl.glmDir + glm, participant_id)
pathSurf = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id)
pathActivity = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id, 'beta')
atlas = 'aparc'
Hem = 'L'

region = 'precentral'

reginfo = pd.read_csv(f'{pathGlm}/{participant_id}_reginfo.tsv', sep='\t')
# reginfo.drop(reginfo[reginfo['cue'] == 'rest'].index, inplace=True)
cue = reginfo.cue.to_list()
run = reginfo.run.to_list()

A = nb.load(os.path.join(pathSurf, f'{participant_id}.{Hem}.32k.{atlas}.label.gii'))
keys = A.darrays[0].data
label = [(l.key, l.label) for l in A.labeltable.labels]
label_df = pd.DataFrame(label, columns=['key', 'label'])
key = label_df.key[label_df.label.to_list().index(region)]

B = nb.load(os.path.join(pathActivity, f'beta.{Hem}.func.gii'))
data = np.array([b.data[keys == key] for b in B.darrays])

dataset = rsa.data.Dataset(
    data,
    channel_descriptors={'channel': np.array(['vertex_' + str(x) for x in range(data.shape[-1])])},
    obs_descriptors={'cue': cue, 'run': run})

# euclidean distance
rdm_eucl = rsa.rdm.calc_rdm_unbalanced(dataset, method='euclidean', descriptor='cue')

# mahalanobis
# noise_diag = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='diag')
# noise_shrink_eye = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_eye')
noise_shrink_diag = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_diag')
# rdm_maha_diag = rsa.rdm.calc_rdm_unbalanced(dataset, method='mahalanobis', descriptor='cue', noise=noise_diag)
# rdm_maha_shrink_eye = rsa.rdm.calc_rdm_unbalanced(dataset, method='mahalanobis', descriptor='cue',
#                                                   noise=noise_shrink_eye)
rdm_maha_shrink_diag = rsa.rdm.calc_rdm_unbalanced(dataset, method='mahalanobis', descriptor='cue',
                                                   noise=noise_shrink_diag)

index = [0, 2, 3, 4, 1, 5]

RDMs = rsa.rdm.RDMs(
    np.concatenate(
        [rdm_eucl.get_matrices()[:, index, :][:, :, index],
         rdm_maha_shrink_diag.get_matrices()[:, index, :][:, :, index]],
        axis=0),
    rdm_descriptors=dict(method=('Euclidean', 'Shrinkage Diagonal')),
    pattern_descriptors={'index': index,
                         'cue': np.unique(cue)[index]})

# visualize
rsa.vis.show_rdm(
    RDMs,
    show_colorbar='panel',
    rdm_descriptor='method',
    pattern_descriptor='cue',
    n_row=1,
    figsize=(15, 5)
)
