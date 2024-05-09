import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import globals as gl
import rsatoolbox as rsa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--participant_id', default='subj101', help='Participant ID')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere')
    parser.add_argument('--glm', default='4', help='GLM model')
    parser.add_argument('--method', default='cv', help='Selected dist')
    # parser.add_argument('--sel_cue', nargs='+', default=['0%', '25%', '50%', '75%', '100%'], help='Selected cue')
    parser.add_argument('--epoch', nargs='+', default=['exec'], help='Selected epoch')
    parser.add_argument('--stimFinger', nargs='+', default=['index', 'ring'], help='Selected stimulated finger')
    parser.add_argument('--instr', nargs='+', default=['go'], help='Selected instruction')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    Hem = args.Hem
    glm = args.glm
    method = args.method
    sel_epoch = args.epoch
    sel_stimFinger = args.stimFinger
    sel_instr = args.instr

    experiment = 'smp1'

    path = os.path.join(gl.baseDir, experiment, gl.RDM, gl.glmDir + glm, participant_id)

    # build filename
    filename = f'{method}.{atlas}.{Hem}'
    if len(sel_epoch) == 1:
        filename += f'.{sel_epoch[0]}'

    if len(sel_instr) == 1:
        filename += f'.{sel_instr[0]}'

    if len(sel_stimFinger) == 1:
        filename += f'.{sel_stimFinger[0]}'

    npz = np.load(os.path.join(path, f'RDMs.{filename}.npz'))
    mat = npz['data_array']
    descr = json.loads(npz['descriptor'].item())

    RDMs = rsa.rdm.RDMs(mat,
                        rdm_descriptors=descr['rdm_descriptors'],
                        pattern_descriptors=descr['pattern_descriptors'])

    if sel_epoch == ['plan']:
        index = np.array([4, 6, 7, 8, 5, 1, 2, 3, 0, 9, 10, 11, 12])
    else:
        index = np.array([1, 2, 3, 0, 4, 5, 6, 7])
    RDMs.reorder(index)

    # visualize
    fig, axs = plt.subplots(2, int(RDMs.n_rdm / 2), figsize=(15, 9), sharex=True, sharey=True)
    for r, rdm in enumerate(RDMs):
        cax = rsa.vis.show_rdm_panel(
            rdm,
            axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))],
            rdm_descriptor='roi',
            cmap='viridis',
            vmin=RDMs.get_matrices().min(),
            vmax=RDMs.get_matrices().max(),
        )

        if sel_epoch == ['plan']:
            axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].axvline(4.5, color='k', lw=.8)
            axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].axvline(8.5, color='k', lw=.8)
            axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].axhline(4.5, color='k', lw=.8)
            axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].axhline(8.5, color='k', lw=.8)
        else:
            axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].axvline(3.5, color='k', lw=.8)
            axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].axhline(3.5, color='k', lw=.8)

        axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].set_xticks(np.linspace(0,
                                                                                          len(RDMs.pattern_descriptors[
                                                                                                  'stimFinger,cue']) - 1,
                                                                                          len(RDMs.pattern_descriptors[
                                                                                                  'stimFinger,cue'])))
        axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].set_xticklabels(
            RDMs.pattern_descriptors['stimFinger,cue'], rotation=45, ha='right')
        axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].set_yticks(
            axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].get_xticks())
        axs[int(r // (RDMs.n_rdm / 2)), int(r % (RDMs.n_rdm / 2))].set_yticklabels(
            RDMs.pattern_descriptors['stimFinger,cue'])

    cbar = fig.colorbar(cax, ax=axs, orientation='horizontal', fraction=.02)
    cbar.set_label('cross-validated multivariate distance (a.u.)')

    fig.suptitle(f'{participant_id}\nepoch:{sel_epoch}, instr:{sel_instr}, stimFinger:{sel_stimFinger}\n')

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id,
                             f'RDMs.{filename}.png'))

    plt.show()
