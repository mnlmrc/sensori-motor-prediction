import argparse
import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
import rsatoolbox as rsa

import mat73
from scipy.io import loadmat

from scipy.linalg import sqrtm

from utils import sort_key

import globals as gl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--participant_id', default='subj100', help='Participant ID (e.g., subj100, subj101, ...)')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--glm', default='5', help='GLM model (e.g., 1, 2, ...)')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    glm = args.glm

    experiment = 'smp1'

    pathROI = os.path.join(gl.baseDir, experiment, gl.ROI, participant_id)
    pathGlm = os.path.join(gl.baseDir, experiment, gl.glmDir + glm, participant_id)

    # Load the ROI file
    print('loading R...')
    R_cell = loadmat(os.path.join(pathROI, f'{participant_id}_{atlas}_region.mat'))['R'][0]
    R = list()

    # load SPM
    print('loading SPM...')
    try:
        SPM = mat73.loadmat(os.path.join(pathGlm, f'SPM.mat'))
        iB = SPM['SPM']['xX']['iB']
    except:
        SPM = loadmat(os.path.join(pathGlm, f'SPM.mat'))
        iB = SPM['SPM'][0][0]['xX']['iB'][0][0][0]

    # load reginfo
    print('loading reginfo...')
    reginfo = pd.read_csv(f'{pathGlm}/{participant_id}_reginfo.tsv', sep='\t')

    for r in R_cell:
        R.append({field: r[field].item() for field in r.dtype.names})

    files = [file for f, file in enumerate(os.listdir(pathGlm)) if file.startswith('beta') and f + 1 < iB[0]]

    # extract betas
    beta = list()
    for f in files:
        print(f'loading {f}...')
        vol = nb.load(os.path.join(pathGlm, f)).get_fdata()
        b = vol
        beta.append(b)
    beta = np.array(beta).reshape(len(files), -1)

    ResMS = nb.load(os.path.join(pathGlm, 'ResMS.nii')).get_fdata().reshape(-1)
    beta_prewhitened = beta / np.sqrt(ResMS)

    # define rois per atlas:
    rois = {
        'Desikan': [
            'rostralmiddlefrontal',
            'caudalmiddlefrontal',
            'precentral',
            'postcentral',
            'superiorparietal',
            'pericalcarine'
        ],
        'BA_handArea': [
            'ba4a', 'ba4p', 'ba3A', 'ba3B', 'ba1', 'ba2'
        ],
        'ROI': [
            'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'
        ]
    }
    rois = rois[atlas]

    # calculate RDMs in each ROI
    RDMs = list()
    for r in R:
        if r["name"] in rois:
            print(f'region:{r["name"]}, hemisphere:{r["hem"]}, {len(r["linvoxidxs"])} voxels')
            # get linear indices
            linvoxidxs = r['linvoxidxs']
            b = beta_prewhitened[:, linvoxidxs].squeeze()
            dataset = rsa.data.Dataset(
                b,
                channel_descriptors={'channel': np.array(['vox_' + str(x) for x in range(b.shape[-1])])},
                obs_descriptors={'conds': reginfo.name,
                                 'run': reginfo.run})
            rdm = rsa.rdm.calc_rdm_unbalanced(dataset, method='crossnobis', descriptor='conds',
                                              cv_descriptor='run')
            rdm.rdm_descriptors = {'roi': r["name"], 'hem': r["hem"], 'index': [0]}
            RDMs.append(rdm)

    RDMs = rsa.rdm.rdms.concat(RDMs)
    RDMs.save(os.path.join(gl.baseDir, experiment, gl.RDM, gl.glmDir + glm, participant_id, f'RDMs.{atlas}.hdf5'),
              file_type='hdf5',
              overwrite=True)
