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

    rois = RDMs.rdm_descriptors['roi']

    dist = RDMs.dissimilarities
    dist_df = pd.DataFrame(dist.T, columns=rois)

    # # check significance
    # t_stats, p_values = ttest_1samp(dist, popmean=0, axis=1)

    fig, axs = plt.subplots(figsize=(4, 5))

    sns.boxplot(data=dist_df, ax=axs, color='gray')
    axs.plot(dist_df.mean(), color='k', marker='o')

    # axs.bar(rois, mdist)
    axs.set_ylabel('multivariate distance (a.u.)')
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right')  # Correct rotation method

    # # Adding asterisks for significant results
    # significance_level = 0.05
    # for idx, p in enumerate(p_values):
    #     if p < significance_level:
    #         axs.text(idx, mdist[idx], '*', ha='center', va='bottom', color='k', fontsize=12)

    fig.subplots_adjust(left=.20, bottom=.25)

    axs.set_title(f'{participant_id}\nepoch:{sel_epoch}, instr:{sel_instr}, stimFinger:{sel_stimFinger}')

    plt.show()

