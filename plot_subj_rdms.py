import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import globals as gl
import rsatoolbox as rsa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--participant_id', default='subj100', help='Participant ID')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere')
    parser.add_argument('--glm', default='1', help='GLM model')
    parser.add_argument('--dist', default='cv', help='Selected dist')
    # parser.add_argument('--sel_cue', nargs='+', default=['0%', '25%', '50%', '75%', '100%'], help='Selected cue')
    parser.add_argument('--epoch', nargs='+', default=['exec', 'plan'], help='Selected epoch')
    parser.add_argument('--stimFinger', nargs='+', default=['index', 'ring', 'none'], help='Selected stimulated finger')
    parser.add_argument('--instr', nargs='+', default=['go', 'nogo'], help='Selected instruction')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    Hem = args.Hem
    glm = args.glm
    dist = args.dist
    sel_epoch = args.epoch
    sel_stimFinger = args.stimFinger
    sel_instr = args.instr

    experiment = 'smp1'

    path = os.path.join(gl.baseDir, experiment, gl.RDM, participant_id)

    # build filename
    filename = f'{dist}.{atlas}.{Hem}'
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

    if sel_stimFinger == ['index']:
        index = [1, 2, 3, 0]
    elif sel_stimFinger == ['ring']:
        index = [0, 1, 2, 3]
    else:
        index = [0, 2, 3, 4, 1]
    RDMs.reorder(index)
    # print(RDMs.pattern_descriptors['cue'])

    # visualize
    fig, axs, oth = rsa.vis.show_rdm(
                    RDMs,
                    show_colorbar=None,
                    rdm_descriptor='roi',
                    pattern_descriptor='cue',
                    n_row=1,
                    figsize=(15, 5),
                    vmin=0, vmax=RDMs.get_matrices().max())

    # oth[-1]['colorbar'].ax.yaxis.set_tick_params(labelleft=True, labelright=False)
    fig.suptitle(f'{participant_id}\nepoch:{sel_epoch}, instr:{sel_instr}, stimFinger:{sel_stimFinger}')
    # fig.tight_layout()

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, f'RDMs.{filename}.png'))

    plt.show()

