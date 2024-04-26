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
# glm = '1'

pathSurf = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id)
pathActivity = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id, 'beta')
atlas = 'aparc'
Hem = 'L'

A = nb.load(os.path.join(pathSurf, f'{participant_id}.{Hem}.32k.{atlas}.label.gii'))
regions = A.darrays[0].data
label = [(l.key, l.label) for l in A.labeltable.labels]
label_df = pd.DataFrame(label, columns=['key', 'label'])

B = nb.load(os.path.join(pathActivity, f'beta.{Hem}.func.gii'))









