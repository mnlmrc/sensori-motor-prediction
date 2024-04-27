import os
import json
import numpy as np
import globals as gl
import rsatoolbox as rsa

participant_id = 'subj100'
experiment = 'smp1'
atlas = 'BA_exvivo'
Hem = 'L'
dist = 'cv'

path = os.path.join(gl.baseDir, experiment, gl.RDM, participant_id)

npz = np.load(os.path.join(path, f'RDMs.{dist}.{atlas}.{Hem}.npz'))
mat = npz['data_array']
descr = json.loads(npz['descriptor'].item())

RDMs = rsa.rdm.RDMs(mat,
                    rdm_descriptors=descr['rdm_descriptors'],
                    pattern_descriptors=descr['pattern_descriptors'])

RDMs.subset_pattern('cue', ['0%', '25%', '50%', '75%', '100%'])
RDMs.n_cond = 5
index = [0, 2, 3, 4, 1]
RDMs.reorder(index)

# visualize
rsa.vis.show_rdm(
    RDMs,
    show_colorbar='figure',
    rdm_descriptor='roi',
    pattern_descriptor='cue',
    n_row=1,
    figsize=(15, 5)
)

