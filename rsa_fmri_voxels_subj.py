import argparse
import json
import os

import nitools as nt
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
    parser.add_argument('--glm', default='9', help='GLM model (e.g., 1, 2, ...)')
    # order:
    # glm8 : [2, 4, 6, 1,  0, 3, 5, 7]
    # glm9 : [0, 4, 7, 10, 2,  5, 8, 11, 3,  1, 6, 9, 12]
    parser.add_argument('--index', nargs='+', type=int, default=[0, 4, 7, 10, 2,  5, 8, 11, 3,  1, 6, 9, 12],
                        help='Label order')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    glm = args.glm
    index = args.index

    experiment = 'smp1'

    pathROI = os.path.join(gl.baseDir, experiment, gl.ROI, participant_id)
    pathGlm = os.path.join(gl.baseDir, experiment, gl.glmDir + glm, participant_id)

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

    # Load the ROI file
    print('loading R...')
    R_cell = loadmat(os.path.join(pathROI, f'{participant_id}_{atlas}_region.mat'))['R'][0]
    R = list()

    for r in R_cell:
        R.append({field: r[field].item() for field in r.dtype.names})

    files = [file for f, file in enumerate(os.listdir(pathGlm)) if file.startswith('beta') and f + 1 < iB[0]]

    ResMS = nb.load(os.path.join(pathGlm, 'ResMS.nii'))

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
            print(f'region:{r["name"]}, hemisphere:{r["hem"]}, {len(r["data"])} voxels')
            beta_prewhitened = list()
            for f in files:
                vol = nb.load(os.path.join(pathGlm, f))
                beta = nt.sample_image(vol, r['data'][:, 0],  r['data'][:, 1], r['data'][:, 2], 0)
                res = nt.sample_image(ResMS, r['data'][:, 0],  r['data'][:, 1], r['data'][:, 2], 0)
                beta_prewhitened.append(beta / np.sqrt(res))

            beta_prewhitened = np.array(beta_prewhitened)
            dataset = rsa.data.Dataset(
                beta_prewhitened,
                channel_descriptors={'channel': np.array(['vox_' + str(x) for x in range(beta_prewhitened.shape[-1])])},
                obs_descriptors={'conds': reginfo.name,
                                 'run': reginfo.run})
            rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run')
            rdm.rdm_descriptors = {'roi': r["name"], 'hem': r["hem"], 'index': [0]}
            rdm.reorder(np.argsort(rdm.pattern_descriptors['conds']))
            rdm.reorder(index)
            RDMs.append(rdm)

    RDMs = rsa.rdm.rdms.concat(RDMs)

    output_path = os.path.join(gl.baseDir, experiment, gl.RDM, gl.glmDir + glm, participant_id)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    RDMs.save(os.path.join(output_path, f'RDMs.vox.{atlas}.hdf5'),
              file_type='hdf5',
              overwrite=True)
