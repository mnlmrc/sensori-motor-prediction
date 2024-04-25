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



