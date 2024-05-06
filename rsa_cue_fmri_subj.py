import argparse
import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
import rsatoolbox as rsa

import globals as gl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument('--participant_id', default='subj101', help='Participant ID (e.g., subj100, subj101, ...)')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere (L/R)')
    parser.add_argument('--glm', default='2', help='GLM model (e.g., 1, 2, ...)')
    # parser.add_argument('--sel_cue', nargs='+', default=['0%', '25%', '50%', '75%', '100%'], help='Selected cue')
    parser.add_argument('--epoch', nargs='+', default=['plan'], help='Selected epoch')
    parser.add_argument('--stimFinger', nargs='+',default=['none'], help='Selected stimulated finger')
    parser.add_argument('--instr', nargs='+', default=['nogo'], help='Selected instruction')

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
    pathRDM = os.path.join(gl.baseDir, experiment, gl.RDM, gl.glmDir + glm, participant_id)
    pathSurf = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id)

    rois = {'ROI': ['PMd',
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

    cues = pd.DataFrame([('0%', 'cue0'),
                         ('25%', 'cue25'),
                         ('50%', 'cue50'),
                         ('75%', 'cue75'),
                         ('100%', 'cue100')],
                        columns=['label', 'instr'])
    map_dict = dict(zip(cues['instr'], cues['label']))
    cue = [map_dict.get(item, item) for item in cue]

    A = nb.load(os.path.join(pathAtlas, f'{atlas}.32k.{Hem}.label.gii'))
    keys = A.darrays[0].data
    label = [(l.key, l.label) for l in A.labeltable.labels[1:]]
    label_df = pd.DataFrame(label, columns=['key', 'label'])

    B = nb.load(os.path.join(pathSurf, f'glm{glm}.beta.{Hem}.func.gii'))
    Res = nb.load(os.path.join(pathSurf, f'glm{glm}.res.{Hem}.func.gii'))

    rdm_eucl = list()
    rdm_maha = list()
    rdm_cv = list()
    for r, roi in enumerate(rois):
        key = label_df.key[label_df.label.to_list().index(roi)]
        print(f'processing {roi}, {(keys == key).sum()} vertices...')

        data = np.array([b.data[(keys == key) & (abs(b.data) > 1e-8)] for b in B.darrays])
        res = Res.darrays[0].data[(keys == key) & (abs(B.darrays[0].data) > 1e-8)]

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

        # # # euclidean distance
        # rdm_eucl.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='euclidean', descriptor='cue'))

        # mahalanobis
        prec = np.linalg.inv(np.diag(res)).astype(float)
        noise = rsa.data.noise.prec_from_unbalanced(dataset, obs_desc='cue', method='shrinkage_diag')
        # rdm_maha.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='mahalanobis', descriptor='cue',
        #                                             noise=prec))
        rdm_cv.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='crossnobis', descriptor='cue',
                                                  noise=noise, cv_descriptor='run'))

    # RDMs_eucl = np.concatenate([rdm.get_matrices() for rdm in rdm_eucl], axis=0)
    # RDMs_maha = np.concatenate([rdm.get_matrices() for rdm in rdm_maha], axis=0)
    RDMs_cv = np.concatenate([rdm.get_matrices() for rdm in rdm_cv], axis=0)

    descr = json.dumps({
        'experiment': experiment,
        'participant': participant_id,
        'rdm_descriptors': dict(roi=tuple(rois)),
        'pattern_descriptors': dict(cue=list(rdm_cv[0].pattern_descriptors['cue']))
    })

    # build filename
    filename = f'{atlas}.{Hem}'

    if len(sel_epoch) == 1:
        filename += f'.{sel_epoch[0]}'

    if len(sel_instr) == 1:
        filename += f'.{sel_instr[0]}'

    if len(sel_stimFinger) == 1:
        filename += f'.{sel_stimFinger[0]}'

    # np.savez(os.path.join(pathRDM, f'RDMs.eucl.{filename}.npz'),
    #          data_array=RDMs_eucl, descriptor=descr, allow_pickle=False)
    # np.savez(os.path.join(pathRDM, f'RDMs.maha.{filename}.npz'),
    #          data_array=RDMs_maha, descriptor=descr, allow_pickle=False)
    np.savez(os.path.join(pathRDM, f'RDMs.cv.{filename}.npz'),
             data_array=RDMs_cv, descriptor=descr, allow_pickle=False)
