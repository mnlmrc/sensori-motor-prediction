import nibabel as nb
import numpy as np

import pandas as pd
import rsatoolbox as rsa
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight

import globals as gl
import os

participant_id = 'subj100'
experiment = 'smp1'
path = os.path.join(gl.baseDir, experiment, gl.glmDir + '1', participant_id)

reginfo = pd.read_csv(f'{path}/{participant_id}_reginfo.tsv', sep='\t')
reginfo.drop(reginfo[reginfo['cue'] == 'rest'].index, inplace=True)

# mask = nb.load(os.path.join(gl.baseDir, experiment, gl.imagingDir, participant_id,'sess1/rmask_gray.nii'))
# mask_array = mask.get_fdata().ravel() > .9

cue = reginfo.cue.to_list()

cue_map = pd.DataFrame([('0%', '000'),
                         ('25%', '025'),
                         ('50%', '050'),
                         ('75%', '075'),
                         ('100%', '100')],
                        columns=['label', 'instr'])
map_dict = dict(zip(cue_map['label'], cue_map['instr']))
cue = np.array([map_dict.get(item, item) for item in cue])

tmp_img = nb.load(path + '/beta_0001.nii')
mask = ~np.isnan(tmp_img.get_fdata())
x, y, z = tmp_img.get_fdata().shape

flattened_data = list()
data = np.zeros((len(reginfo), x, y, z))
blocks = np.zeros(len(reginfo))
for i, im in enumerate(reginfo.iterrows()):
    print('processing: beta_%04d.nii' % (im[0] + 1))
    data[i] = nb.load(path + '/beta_%04d.nii' % (im[0] + 1)).get_fdata()
    # data_array = data.get_fdata().ravel()
    # flattened_data.append(data_array[mask_array])
    blocks[i] = im[1].run

centers, neighbors = get_volume_searchlight(mask, radius=5, threshold=1)

data_2d = data.reshape([data.shape[0], -1])
data_2d = np.nan_to_num(data_2d)

rdm_maha = list()
rdm_eucl = list()
for n, neigh in enumerate(neighbors):
    print('%d complete...' % np.ceil(100 * n/len(neighbors)).astype(int))

    # create dataset
    dataset = rsa.data.Dataset(
        data_2d[:, neigh],
        channel_descriptors={'channel': np.array(['center_' + str(x) for x in range(len(neigh))])},
        obs_descriptors={'cue': cue, 'blocks': blocks})

    # euclidean distance
    rdm_eucl.append(rsa.rdm.calc_rdm(dataset, method='euclidean', descriptor='cue'))

    # mahalanobis
    # noise_prec_diag = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='diag')
    # noise_prec_shrink_eye = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_eye')
    noise_prec_shrink_diag = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_diag')
    rdm_maha.append(rsa.rdm.calc_rdm(dataset, method='mahalanobis', descriptor='cue', noise=noise_prec_shrink_diag))


# flattened_data_array = np.array(flattened_data)
#
# nVox = flattened_data_array.shape[-1]
#
# cue = np.array(cue)
# blocks = np.array(blocks)



#
# rdm_eucl = rsa.rdm.calc_rdm(dataset, method='euclidean', descriptor='cue')
# rdm_maha = rsa.rdm.calc_rdm(dataset, method='mahalanobis', descriptor='cue', noise=)

# rdm_conc = rsa.rdm.RDMs(
#         np.concatenate(
#             [rdm_eucl.get_matrices(),
#              rdm_maha.get_matrices()],
#             axis=0),
#         rdm_descriptors=dict(method=('Euclidean', 'Mahalanobis')),
#         pattern_descriptors={'cue': cues.label.to_list() + ['rest']}
#     )
#
# fig, axs, _ = rsa.vis.show_rdm(rdm_maha,
#     show_colorbar='panel',
#     rdm_descriptor='method',
#     pattern_descriptor='cue',
#     cmap='jet',
#     n_row=1,
#     figsize=(8, 5)
# )
# fig = rsa.vis.show_MDS(rdm_maha, pattern_descriptor='cue')

