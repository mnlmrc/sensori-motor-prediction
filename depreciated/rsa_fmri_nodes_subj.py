import argparse
import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
import rsatoolbox as rsa

from scipy.linalg import sqrtm

from utils import sort_key

import globals as gl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument('--participant_id', default='subj101', help='Participant ID (e.g., subj100, subj101, ...)')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere (L/R)')
    parser.add_argument('--glm', default='8', help='GLM model (e.g., 1, 2, ...)')
    parser.add_argument('--index', nargs='+', type=int, default=[2, 4, 6, 1,  0, 3, 5, 7],
                        help='Label order')
    # parser.add_argument('--sel_cue', nargs='+', default=['0%', '25%', '50%', '75%', '100%'], help='Selected cue')
    # parser.add_argument('--condition', nargs='+', default=['plan'], help='Selected epoch')
    # parser.add_argument('--stimFinger', nargs='+',default=['none'], help='Selected stimulated finger')
    # parser.add_argument('--instr', nargs='+', default=['nogo', 'go'], help='Selected instruction')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    Hem = args.Hem
    glm = args.glm
    index = args.index
    # sel_epoch = args.condition
    # sel_stimFinger = args.stimFinger
    # sel_instr = args.instr

    experiment = 'smp1'

    pathGlm = os.path.join(gl.baseDir, experiment, gl.glmDir + glm, participant_id)
    pathAtlas = os.path.join(gl.baseDir, experiment, 'atlases')
    pathRDM = os.path.join(gl.baseDir, experiment, gl.RDM, gl.glmDir + glm, participant_id)
    pathSurf = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id)

    rois = {'ROI': ['SMA',
                    'PMd',
                    'PMv',
                    'M1',
                    'S1',
                    'SPLa',
                    'SPLp',
                    'V1']}
    rois = rois[atlas]

    reginfo = pd.read_csv(f'{pathGlm}/{participant_id}_reginfo.tsv', sep='\t')

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
        res = Res.darrays[0].data[(keys == key) & (abs(B.darrays[0].data) > 1e-8)].astype(float)
        # res = np.diag(res)

        # Euclidean distance (prewhitened)
        data_prewhitened = data / np.sqrt(res)
        dataset = rsa.data.Dataset(
            data_prewhitened,
            channel_descriptors={'channel': np.array(['vertex_' + str(x) for x in range(data.shape[-1])])},
            obs_descriptors={'conds': reginfo.name,
                             'run': reginfo.run})
        # rdm_eucl.append(rsa.rdm.calc_rdm_unbalanced(dataset_prewhitened,
        #                                             method='euclidean',
        #                                             descriptor='conds'))

        # mahalanobis (non-prewhitened)
        # dataset = rsa.data.Dataset(
        #     data,
        #     channel_descriptors={'channel': np.array(['vertex_' + str(x) for x in range(data.shape[-1])])},
        #     obs_descriptors={'conds': reginfo.name,
        #                      'run': reginfo.run})
        # rdm_maha.append(rsa.rdm.calc_rdm_unbalanced(dataset, method='mahalanobis', descriptor='conds',
        #                                             noise=np.linalg.inv(res)))
        # noise = rsa.data.noise.prec_from_measurements(dataset, obs_desc='conds', method='shrinkage_eye')
        rdm = rsa.rdm.calc_rdm_unbalanced(dataset, method='crossnobis', descriptor='conds',
                                                    cv_descriptor='run')
        rdm.reorder(np.argsort(rdm.pattern_descriptors['conds']))
        rdm.reorder(index)
        rdm_cv.append(rdm)


    # RDMs_eucl = rsa.rdm.RDMs(np.concatenate([rdm.get_matrices() for rdm in rdm_eucl], axis=0),
    #                          dissimilarity_measure='euclidean',
    #                          rdm_descriptors={'ROI': rois},
    #                          pattern_descriptors=rdm_eucl[0].pattern_descriptors)
    #
    # RDMs_maha = rsa.rdm.RDMs(np.concatenate([rdm.get_matrices() for rdm in rdm_maha], axis=0),
    #                          dissimilarity_measure='mahalanobis',
    #                          rdm_descriptors={'ROI': rois},
    #                          pattern_descriptors=rdm_maha[0].pattern_descriptors)

    RDMs_cv = rsa.rdm.RDMs(np.concatenate([rdm.get_matrices() for rdm in rdm_cv], axis=0),
                             dissimilarity_measure='crossnobis',
                             rdm_descriptors={'roi': rois},
                             pattern_descriptors=rdm_cv[0].pattern_descriptors)



    # rdm_maha = rsa.rdm.concat(rdm_maha)
    # rdm_maha.reorder(rdm_maha.pattern_descriptors['stimFinger,cue'].argsort())
    # RDMs_maha = np.concatenate([rdm.get_matrices() for rdm in rdm_maha], axis=0)
    #
    # rdm_cv = rsa.rdm.concat(rdm_cv)
    # rdm_cv.reorder(rdm_cv.pattern_descriptors['stimFinger,cue'].argsort())
    # RDMs_cv = np.concatenate([rdm.get_matrices() for rdm in rdm_cv], axis=0)
    #
    descr = json.dumps({
        'experiment': experiment,
        'participant': participant_id,
        'rdm_descriptors': dict(roi=tuple(rois)),
        'pattern_descriptors': {'conds': list(rdm_cv[0].pattern_descriptors['conds'])}
    })

    # if len(sel_epoch) == 1:
    #     filename += f'.{sel_epoch[0]}'
    #
    # if len(sel_instr) == 1:
    #     filename += f'.{sel_instr[0]}'
    #
    # if len(sel_stimFinger) == 1:
    #     filename += f'.{sel_stimFinger[0]}'
    #
    # np.savez(os.path.join(pathRDM, f'RDMs.eucl.{filename}.npz'),
    #          data_array=RDMs_eucl, descriptor=descr, allow_pickle=False)
    # # np.savez(os.path.join(pathRDM, f'RDMs.maha.{filename}.npz'),
    # #          data_array=RDMs_maha, descriptor=descr, allow_pickle=False)
    # np.savez(os.path.join(pathRDM, f'RDMs.surf.cv.{filename}.npz'),
    #          data_array=RDMs_cv, descriptor=descr, allow_pickle=False)

    RDMs_cv.save(os.path.join(pathRDM, f'RDMs.surf.{atlas}.{Hem}.hdf5'),
              file_type='hdf5',
              overwrite=True)
