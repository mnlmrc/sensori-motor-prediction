import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import globals as gl
import rsatoolbox as rsa

if __name__ == "__main__":

    # order:
    # glm9 : [0, 4, 7, 10, 2,  5, 8, 11, 3,  1, 6, 9, 12]
    # Argument parsing
    parser = argparse.ArgumentParser(description="Plot RDM")
    parser.add_argument('--participant_id', default='subj106', help='Participant ID')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--glm', default='10', help='GLM model')
    parser.add_argument('--type', default='voxels', help='GLM model')
    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    glm = args.glm
    type = args.type

    experiment = 'smp1'
    path = os.path.join(gl.baseDir, experiment, gl.RDM, gl.glmDir + glm, participant_id)

    # Load RDMs
    RDMs = rsa.rdm.load_rdm(os.path.join(path, f'RDMs.vox.{atlas}.hdf5'))
    # if index is not None:
    #     RDMs.reorder(np.argsort(RDMs.pattern_descriptors['conds']))
    #     RDMs.reorder(index)
    # RDMs.pattern_descriptors['conds'] = [c.decode('utf-8').replace(' ', '') for c in RDMs.pattern_descriptors['conds']]

    vmin = RDMs.dissimilarities.min()
    vmax = 1 #RDMs.dissimilarities.max()

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
        fig.suptitle(f'RDMs, {participant_id}, glm{glm}, hemisphere: {hem}')

        fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, f'RDMs.{atlas}.glm{glm}.{hem}.png'), dpi=300)

    plt.show()
