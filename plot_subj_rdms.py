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
    parser.add_argument('--atlas', default='aparc', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere')
    parser.add_argument('--glm', default='1', help='GLM model')
    parser.add_argument('--dist', default='cv', help='Selected cue')
    parser.add_argument('--epoch', default='plan', help='Selected epoch')
    parser.add_argument('--stimFinger', default='none', help='Selected stimulated finger')
    parser.add_argument('--instr', default='nogo', help='Selected instruction')

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

    npz = np.load(os.path.join(path, f'RDMs.{dist}.{atlas}.{Hem}.{sel_epoch}.{sel_instr}.{sel_stimFinger}.npz'))
    mat = npz['data_array']
    descr = json.loads(npz['descriptor'].item())

    RDMs = rsa.rdm.RDMs(mat,
                        rdm_descriptors=descr['rdm_descriptors'],
                        pattern_descriptors=descr['pattern_descriptors'])

    RDMs.subset_pattern('cue', ['0%', '25%', '50%', '75%', '100%'])
    RDMs.n_cond = 5
    index = [0, 2, 3, 4, 1]
    RDMs.reorder(index)

    # visualize
    fig, axs, oth = rsa.vis.show_rdm(
                    RDMs,
                    show_colorbar=None,
                    rdm_descriptor='roi',
                    pattern_descriptor='cue',
                    n_row=1,
                    figsize=(15, 5),
                    vmin=0, vmax=.06)

    # oth[-1]['colorbar'].ax.yaxis.set_tick_params(labelleft=True, labelright=False)
    fig.suptitle(f'{participant_id}\nepoch:{sel_epoch}, instr:{sel_instr}, stimFinger:{sel_stimFinger}')
    # fig.tight_layout()

    plt.show()

