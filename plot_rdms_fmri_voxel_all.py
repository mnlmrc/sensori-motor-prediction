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
        'subj104',
        'subj105',
        'subj106'
    ], help='Participant IDs')
    parser.add_argument('--atlas', default='ROI', help='atlas')
    parser.add_argument('--glm', default='10', help='glm')
    parser.add_argument('--rois', nargs='+',default=['PMd', 'PMv', 'M1', 'S1',], help='glm')

    args = parser.parse_args()
    participants = args.participants
    atlas = args.atlas
    glm = args.glm
    rois = args.rois
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

    RDMs.pattern_descriptors['conds'] = [c.replace(' ', '') for c in RDMs.pattern_descriptors['conds']]

    RDMs = RDMs.subsample(by='roi', value=rois)

    if RDMs.n_rdm <= 8:
        num_of_rows = 1
    elif (RDMs.n_rdm > 8) &  RDMs.n_rdm <= 16:
        num_of_rows = 2
    elif (RDMs.n_rdm > 16) & (RDMs.n_rdm <= 24):
        num_of_rows = 3
    else:
        num_of_rows = 4

    # planning
    hemispheres = ['L', 'R']
    for hem in hemispheres:

        rdms = RDMs.subsample_pattern(by='conds', value=['0%', '25%', '50%', '75%', '100%'])

        vmin = rdms.dissimilarities.min()
        vmax = rdms.dissimilarities.max()

        rdms = rdms.subset(by='hem', value=hem)

        fig, axs = plt.subplots(num_of_rows, rdms.n_rdm // num_of_rows, figsize=(8, 5), sharex=True, sharey=True)

        for r, rdm in enumerate(rdms):
            try:
                ax = axs[r // (rdms.n_rdm // num_of_rows), r % (rdms.n_rdm // num_of_rows)]
            except:
                ax = axs[r % (rdms.n_rdm // num_of_rows)]

            cax = rsa.vis.show_rdm_panel(
                rdm, ax, rdm_descriptor='roi', cmap='viridis', vmin=vmin, vmax=vmax
            )
            ax.set_xticks(np.arange(len(rdms.pattern_descriptors['conds'])))
            ax.set_xticklabels(rdms.pattern_descriptors['conds'], rotation=45, ha='right')
            ax.set_yticks(ax.get_xticks())
            ax.set_yticklabels(rdms.pattern_descriptors['conds'])

        cbar = fig.colorbar(cax, ax=axs, orientation='horizontal', fraction=.02)
        cbar.set_label('Cross-validated multivariate distance (a.u.)')
        fig.suptitle(f'RDMs, all participants (N={len(participants)}), glm{glm}, hemisphere: {hem}')

        fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', f'RDMs.{hem}.plan.svg'))

        # for r, rdm in enumerate(rdms):
        #     rsa.vis.scatter_plot.show_MDS(rdm, pattern_descriptor='conds')
        #     plt.xlim([-.05, .05])
        #     plt.ylim([-.05, .05])
        #     plt.title(f'{rdm.rdm_descriptors["roi"], rdm.rdm_descriptors["hem"]}')
        #     # plt.tick_params(axis='both', which='both', bottom=True, top=False,
        #     #                right=False, left=True, labelbottom=True, labeltop=False,
        #     #                labelleft=True, labelright=False)

    plt.show()

    # execution
    for hem in hemispheres:

        rdms = RDMs.subsample_pattern(by='conds', value=['0%,ring', '25%,ring', '50%,ring', '75%,ring',
                                                         '25%,index', '50%,index', '75%,index', '100%,index'])

        vmin = rdms.dissimilarities.min()
        vmax = rdms.dissimilarities.max()

        rdms = rdms.subset(by='hem', value=hem)

        fig, axs = plt.subplots(num_of_rows, rdms.n_rdm // num_of_rows, figsize=(15, 9), sharex=True, sharey=True)

        for r, rdm in enumerate(rdms):
            try:
                ax = axs[r // (rdms.n_rdm // num_of_rows), r % (rdms.n_rdm // num_of_rows)]
            except:
                ax = axs[r % (rdms.n_rdm // num_of_rows)]
            cax = rsa.vis.show_rdm_panel(
                rdm, ax, rdm_descriptor='roi', cmap='viridis', vmin=vmin, vmax=vmax
            )
            ax.set_xticks(np.arange(len(rdms.pattern_descriptors['conds'])))
            ax.set_xticklabels(rdms.pattern_descriptors['conds'], rotation=45, ha='right')
            ax.set_yticks(ax.get_xticks())
            ax.set_yticklabels(rdms.pattern_descriptors['conds'])

        cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=.01)
        cbar.set_label('Cross-validated multivariate distance (a.u.)')
        fig.suptitle(f'RDMs, all participants (N={len(participants)}), glm{glm}, hemisphere: {hem}')


        fig.savefig(os.path.join(gl.baseDir, experiment,'figures', f'RDMs.{hem}.exec.svg'))

    plt.show()


