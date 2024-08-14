import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp

import globals as gl

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--experiment', default='smp0', help='')
    parser.add_argument('--participants', default=['subj100',
                                                   'subj101',
                                                   'subj102',
                                                   'subj103',
                                                   'subj104',
                                                   'subj105',
                                                   'subj106',
                                                   'subj107',
                                                   'subj108',
                                                   'subj109',
                                                   'subj110'
                                                   ], help='')

    args = parser.parse_args()

    experiment = args.experiment
    participant_ids = args.participants

    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')

    fsample = 2148
    prestim = int(1 * fsample)
    poststim = int(2 * fsample)
    win = [(prestim - int(.5 * fsample), prestim),
           (prestim + int(.025 * fsample), prestim + int(.05 * fsample)),
           (prestim + int(.05 * fsample), prestim + int(.1 * fsample)),
           (prestim + int(.1 * fsample), prestim + int(.5 * fsample))]

    dist = list()
    for participant in participant_ids:
        sn = int(''.join([c for c in participant if c.isdigit()]))

        print(f'loading {participant}...')

        npz = np.load(os.path.join(gl.baseDir, experiment, participant, 'mov', f'{experiment}_{sn}_distances.npz'))
        dist.append(npz['data_array'])
        descr = json.loads(npz['descriptor'].item())
        # win_size = descr['win_size']


    dist = np.array(dist)

    timeAx = (np.linspace(-1, 2 , dist.shape[-1]) -
              latency[['ring', 'index']].mean(axis=1).to_numpy())

    fig, axs = plt.subplots()

    colors = ['purple', 'darkorange']
    labels = ['finger', 'cue', 'interaction']
    sig_lev = [1.7, 1.75, 1.8]

    t_res = ttest_1samp(dist, 0, axis=0, alternative='greater')
    pval = t_res.pvalue < .05  # corrected with fdr remain just below 0.06
    pval_diff = np.diff(np.concatenate([pval.astype(int), np.zeros((3, 1))], axis=1))

    for i in range(2): #range(dist.shape[1]):
        y = dist.mean(axis=0)[i]
        yerr = dist.std(axis=0)[i] / np.sqrt(dist.shape[0])
        axs.plot(timeAx, y, label=labels[i], color=colors[i])
        axs.fill_between(timeAx, y - yerr, y + yerr, color=colors[i], alpha=.3, lw=0)


    axs.set_xlabel('time relative to stimulation (s)', fontsize=12)
    axs.set_ylabel('cross-validated multivariate distance (a.u.)', fontsize=12)

    axs.axvline(0, color='k', ls='-', lw=.8)
    axs.axvline(.2, color='k', ls=':', lw=.8)
    axs.axvline(.4, color='k', ls='--', lw=.8)
    axs.axvline(1, color='k', ls='-.', lw=.8)
    axs.axhline(0, color='k', ls='-', lw=.8)

    axs.set_xlim([-1, 1])
    # axs.set_ylim([0, 1.2])
    axs.set_yscale('symlog', linthresh=.1)
    axs.axvline(0, color='k', lw=.8)
    axs.spines[['top', 'right']].set_visible(False)

    fig.legend(ncols=1, loc='upper right', ncol=2, fontsize=12)

    # axs.set_title('cross-validated distance over time', fontsize=16)

    # ori = [-.0375, .025, .0625, .125]
    # tit = ['Pre', 'SLR', 'LLR', 'Vol']
    # for w, o in enumerate(ori):
    axin = axs.inset_axes([-.75, .15, .5, 8], transform=axs.transData, zorder=-1)
    axin.hist(dist[:, 1, 400], color='darkorange', alpha=.2, bins=4)
    axin.hist(dist[:, 0, 400], color='purple', alpha=.2, bins=4)

    axin.set_ylabel('no. participants', fontsize=8)
    axin.set_xlabel('multivariate distance (a.u.)', fontsize=8)
    axin.set_title('across participant distribution\nof finger and cue effect\nbefore perturbation', fontsize=10)



    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'dist.timec.force.svg'))

    plt.show()
