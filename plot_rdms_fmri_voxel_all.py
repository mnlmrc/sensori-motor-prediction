import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import globals as gl
import rsatoolbox as rsa

from scipy.spatial.distance import squareform

from matplotlib.patches import Polygon

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--participants', default=[
        'subj100',
        'subj101',
        'subj102',
        'subj103',
        # 'subj105',
        'subj106'
    ], help='Participant IDs')
    parser.add_argument('--atlas', default='ROI', help='atlas')
    parser.add_argument('--glm', default='9', help='glm')

    args = parser.parse_args()
    participants = args.participants
    atlas = args.atlas
    glm = args.glm
    # index = args.index

    experiment = 'smp1'

    path = os.path.join(gl.baseDir, experiment, gl.RDM, gl.glmDir + glm)

    rdms = list()
    for p, participant in enumerate(participants):
        # Load RDMs
        rdm = rsa.rdm.load_rdm(os.path.join(path, participant, f'RDMs.vox.{atlas}.hdf5'))
        # rdm.reorder(np.argsort(rdm.pattern_descriptors['conds']))
        # rdm.reorder(index)
        rdms.append(rdm.get_matrices())

    rdms = np.array(rdms)
    RDMs = rsa.rdm.rdms.RDMs(rdms.mean(axis=0),
                             dissimilarity_measure='crossnobis',
                             descriptors={},
                             rdm_descriptors=rdm.rdm_descriptors,
                             pattern_descriptors=rdm.pattern_descriptors)

    # RDMs.pattern_descriptors['conds'] = [c.decode('utf-8').replace(' ', '') for c in RDMs.pattern_descriptors['conds']]

    vmin = 0  # RDMs.dissimilarities.min()
    vmax = .5  # RDMs.dissimilarities.max()

    if RDMs.n_rdm <= 16:
        num_of_rows = 2
    elif (RDMs.n_rdm > 16) & (RDMs.n_rdm <= 24):
        num_of_rows = 3
    else:
        num_of_rows = 4

    hemispheres = ['L', 'R']
    for hem in hemispheres:
        rdms = RDMs.subset(by='hem', value=hem)
        fig, axs = plt.subplots(num_of_rows, rdms.n_rdm // num_of_rows, figsize=(15, 9), sharex=True, sharey=True)

        for r, rdm in enumerate(rdms):
            ax = axs[r // (rdms.n_rdm // num_of_rows), r % (rdms.n_rdm // num_of_rows)]
            cax = rsa.vis.show_rdm_panel(
                rdm, ax, rdm_descriptor='roi', cmap='viridis', vmin=vmin, vmax=vmax
            )
            ax.set_xticks(np.arange(len(RDMs.pattern_descriptors['conds'])))
            ax.set_xticklabels(RDMs.pattern_descriptors['conds'], rotation=45, ha='right')
            ax.set_yticks(ax.get_xticks())
            ax.set_yticklabels(RDMs.pattern_descriptors['conds'])

        cbar = fig.colorbar(cax, ax=axs, orientation='horizontal', fraction=.02)
        cbar.set_label('Cross-validated multivariate distance (a.u.)')
        fig.suptitle(f'RDMs, all participants (N={len(participants)}), glm{glm}, hemisphere: {hem}')

    plt.show()
