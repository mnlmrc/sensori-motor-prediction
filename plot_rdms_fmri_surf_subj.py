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
    parser.add_argument('--participant_id', default='subj100', help='Participant ID')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--glm', default='8', help='GLM model')
    parser.add_argument('--type', default='surf', help='GLM model')
    parser.add_argument('--hem', default='L', help='GLM model')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    glm = args.glm
    Hem = args.hem

    experiment = 'smp1'
    path = os.path.join(gl.baseDir, experiment, gl.RDM, gl.glmDir + glm, participant_id)

    # Load RDMs
    rdms = rsa.rdm.load_rdm(os.path.join(path, f'RDMs.surf.{atlas}.{Hem}.hdf5'))
    # if index is not None:
    #     RDMs.reorder(np.argsort(RDMs.pattern_descriptors['conds']))
    #     RDMs.reorder(index)
    # RDMs.pattern_descriptors['conds'] = [c.decode('utf-8').replace(' ', '') for c in RDMs.pattern_descriptors['conds']]

    vmin = rdms.dissimilarities.min()
    vmax = rdms.dissimilarities.max()

    if rdms.n_rdm <= 16:
        num_of_rows = 2
    elif (rdms.n_rdm > 16) & (rdms.n_rdm <= 24):
        num_of_rows = 3
    else:
        num_of_rows = 4


    fig, axs = plt.subplots(num_of_rows, rdms.n_rdm // num_of_rows, figsize=(15, 9), sharex=True, sharey=True)

    for r, rdm in enumerate(rdms):
        ax = axs[r // (rdms.n_rdm // num_of_rows), r % (rdms.n_rdm // num_of_rows)]
        cax = rsa.vis.show_rdm_panel(
            rdm, ax, rdm_descriptor='roi', cmap='viridis', vmin=vmin, vmax=vmax
        )
        ax.set_xticks(np.arange(len(rdms.pattern_descriptors['conds'])))
        ax.set_xticklabels(rdms.pattern_descriptors['conds'], rotation=45, ha='right')
        ax.set_yticks(ax.get_xticks())
        ax.set_yticklabels(rdms.pattern_descriptors['conds'])

    cbar = fig.colorbar(cax, ax=axs, orientation='horizontal', fraction=.02)
    cbar.set_label('Cross-validated multivariate distance (a.u.)')
    fig.suptitle(f'RDMs, {participant_id}, glm{glm}, hemisphere: {Hem}')

    # fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, f'RDMs.{atlas}.glm{glm}.{hem}.png'), dpi=300)

    plt.show()
