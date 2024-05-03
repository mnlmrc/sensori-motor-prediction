import argparse
import os
import json

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_1samp
import numpy as np
import globals as gl
import rsatoolbox as rsa

import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--participant_id', default='subj100', help='Participant ID')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere')
    parser.add_argument('--glm', default='1', help='GLM model')
    parser.add_argument('--dist', default='cv', help='Selected cue')
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

    RDMs = RDMs.subset_pattern('cue', ['0%', '25%', '50%', '75%', '100%'])
    # RDMs.n_cond = 5

    rois = RDMs.rdm_descriptors['roi']

    dist = RDMs.dissimilarities
    dist_df = pd.DataFrame(dist.T, columns=rois)

    # # check significance
    # t_stats, p_values = ttest_1samp(dist, popmean=0, axis=1)

    fig, axs = plt.subplots(figsize=(7, 5))

    sns.boxplot(data=dist_df, ax=axs, color='gray', width=.5)
    axs.plot(dist_df.mean(), color='k', marker='o')

    # axs.bar(rois, mdist)
    axs.set_ylabel('multivariate distance (a.u.)')
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right')  # Correct rotation method

    axs.set_ylim([0, .5])

    axs.set_title(f'{participant_id}\nepoch:{sel_epoch}, instr:{sel_instr}, stimFinger:{sel_stimFinger}\n')

    axs.set_yscale('log')

    # # Adding asterisks for significant results
    # significance_level = 0.05
    # for idx, p in enumerate(p_values):
    #     if p < significance_level:
    #         axs.text(idx, mdist[idx], '*', ha='center', va='bottom', color='k', fontsize=12)

    fig.subplots_adjust(left=.3, bottom=.10, right=.7, top=.85)

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, f'dist.{filename}.png'))

    plt.show()

