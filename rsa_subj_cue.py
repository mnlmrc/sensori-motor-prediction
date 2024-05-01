import argparse
import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
import rsatoolbox as rsa

import globals as gl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--participant_id', default='subj101', help='Participant ID')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere')
    parser.add_argument('--glm', default='1', help='GLM model')
    # parser.add_argument('--sel_cue', nargs='+', default=['0%', '25%', '50%', '75%', '100%'], help='Selected cue')
    parser.add_argument('--epoch', default='exec', help='Selected epoch')
    parser.add_argument('--stimFinger', default='ring', help='Selected stimulated finger')
    parser.add_argument('--instr', default='go', help='Selected instruction')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    Hem = args.Hem
    glm = args.glm
    sel_epoch = args.epoch
    sel_stimFinger = args.stimFinger
    sel_instr = args.instr

    experiment = 'smp1'

    pathGlm = os.path.join(gl.baseDir, experiment, gl.glmDir + glm, participant_id)
    pathAtlas = os.path.join(gl.baseDir, experiment, 'atlases')
    pathRDM = os.path.join(gl.baseDir, experiment, gl.RDM, participant_id)
    pathActivity = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id, 'beta')

    rois = {'BA_exvivo': ['BA1_exvivo',
                          'BA2_exvivo',
                          'BA3a_exvivo',
                          'BA3b_exvivo',
                          'BA4a_exvivo',
                          'BA4p_exvivo',
                          'BA6_exvivo'],
            'aparc': ['precentral',
                      'postcentral',
                      'caudalmiddlefrontal',
                      'pericalcarine'],
            'ROI': ['PMd',
                    'PMv',
                    'M1',
                    'S1',
                    'SPLa',
                    'SPLp',
                    'V1']}
    rois = rois[atlas]

    reginfo = pd.read_csv(f'{pathGlm}/{participant_id}_reginfo.tsv', sep='\t')
    for col in reginfo.columns:
        reginfo[col] = reginfo[col].astype(str).str.replace(' ', '', regex=True)  # remove spaces from dataframe
    cue = reginfo.cue.to_list()
    stimFinger = reginfo.stimFinger.to_list()
    epoch = reginfo.epoch.to_list()
    instr = reginfo.instr.to_list()
    run = reginfo.run.to_list()

    A = nb.load(os.path.join(pathAtlas, f'{atlas}.32k.{Hem}.label.gii'))
    keys = A.darrays[0].data
    label = [(l.key, l.label) for l in A.labeltable.labels[1:]]
    label_df = pd.DataFrame(label, columns=['key', 'label'])

    B = nb.load(os.path.join(pathActivity, f'beta.{Hem}.func.gii'))

    rdm_eucl = list()
    rdm_maha = list()
    rdm_cv = list()
    for r, roi in enumerate(rois):
        key = label_df.key[label_df.label.to_list().index(roi)]
        print(f'processing {roi}, {(keys == key).sum()} vertices...')

        data = np.array([b.data[(keys == key) & (abs(b.data) > 1e-8)] for b in B.darrays])

        dataset = rsa.data.Dataset(
            data,
            channel_descriptors={'channel': np.array(['vertex_' + str(x) for x in range(data.shape[-1])])},
            obs_descriptors={'cue': cue,
                             'stimFinger': stimFinger,
                             'epoch': epoch,
                             'instr': instr,
                             'run': run})
        # dataset.subset_obs('cue', sel_cue)
        dataset = dataset.subset_obs('stimFinger', sel_stimFinger)
        dataset = dataset.subset_obs('epoch', sel_epoch)
        dataset = dataset.subset_obs('instr', sel_instr)

        # euclidean distance
        rdm_eucl.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='euclidean', descriptor='cue'))

        # mahalanobis
        noise = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_diag')
        rdm_maha.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='mahalanobis', descriptor='cue',
                                                    noise=noise))
        rdm_cv.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='crossnobis', descriptor='cue',
                                                  noise=noise, cv_descriptor='run'))

    RDMs_eucl = np.concatenate([rdm.get_matrices() for rdm in rdm_eucl], axis=0)
    RDMs_maha = np.concatenate([rdm.get_matrices() for rdm in rdm_maha], axis=0)
    RDMs_cv = np.concatenate([rdm.get_matrices() for rdm in rdm_cv], axis=0)

    descr = json.dumps({
        'experiment': experiment,
        'participant': participant_id,
        'rdm_descriptors': dict(roi=tuple(rois)),
        'pattern_descriptors': dict(cue=list(rdm_cv[0].pattern_descriptors['cue']))
    })

    np.savez(os.path.join(pathRDM, f'RDMs.eucl.{atlas}.{Hem}.{sel_epoch}.{sel_instr}.{sel_stimFinger}.npz'),
             data_array=RDMs_eucl, descriptor=descr, allow_pickle=False)
    np.savez(os.path.join(pathRDM, f'RDMs.maha.{atlas}.{Hem}.{sel_epoch}.{sel_instr}.{sel_stimFinger}.npz'),
             data_array=RDMs_maha, descriptor=descr, allow_pickle=False)
    np.savez(os.path.join(pathRDM, f'RDMs.cv.{atlas}.{Hem}.{sel_epoch}.{sel_instr}.{sel_stimFinger}.npz'),
             data_array=RDMs_cv, descriptor=descr, allow_pickle=False)
