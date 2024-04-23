import nibabel as nb
import numpy as np

import pandas as pd
import rsatoolbox as rsa

import globals as gl
import os

subj_id = 'subj100'
experiment = 'smp1'
path = os.path.join(gl.baseDir, experiment, gl.glmDir + '1', subj_id)

reginfo = pd.read_csv(f'{path}/{subj_id}_reginfo.tsv', sep='\t')

cue = reginfo.cue.to_list()

cues = pd.DataFrame([('0%', '000'),
                         ('25%', '025'),
                         ('50%', '050'),
                         ('75%', '075'),
                         ('100%', '100')],
                        columns=['label', 'instr'])
map_dict = dict(zip(cues['label'], cues['instr']))
cue = [map_dict.get(item, item) for item in cue]

flattened_data = list()
blocks = list()
for im in range(len(reginfo)):
    print('processing: beta_%04d.nii' % (im + 1))
    data = nb.load(path + '/beta_%04d.nii' % (im + 1))
    data_array = data.get_fdata()
    flattened_data.append(data_array.ravel())
    blocks.append(reginfo.run[im])

flattened_data_array = np.nan_to_num(np.array(flattened_data))

nVox = flattened_data_array.shape[-1]

cue = np.array(cue)
blocks = np.array(blocks)

dataset = rsa.data.Dataset(
    flattened_data_array,
    channel_descriptors={'channel': np.array(['voxel_' + str(x) for x in np.arange(nVox)])},
    obs_descriptors={'cue': cue, 'blocks': blocks})

rdm_eu = rsa.rdm.calc_rdm(dataset, method='euclidean', descriptor='cue')
rdm_cv = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='cue', cv_descriptor='blocks', )

fig, axs, ret_val = rsa.vis.show_rdm(rdm_eu)

