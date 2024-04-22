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

# img = '/beta_0001.nii'

maxRegr = 10

cue = reginfo.cue[:maxRegr]
cue = cue.to_list()

flattened_data = list()
for im in range(1, maxRegr + 1):
    print('processing: beta_%04d.nii' % im)
    data = nb.load(path + '/beta_%04d.nii' % im)
    data_array = data.get_fdata()
    flattened_data.append(data_array.ravel())

flattened_data_array = np.array(flattened_data)

nVox = flattened_data_array.shape[-1]

dataset = rsa.data.Dataset(
    flattened_data_array,
    channel_descriptors={'channel': np.array(['voxel_' + str(x) for x in np.arange(nVox)])},
    obs_descriptors={'cue': cue})

