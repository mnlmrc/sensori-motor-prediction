import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

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
                                                   'subj110'], help='')

    args = parser.parse_args()

    experiment = args.experiment
    participant_ids = args.participants

    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')

    dist = list()
    for participant in participant_ids:
        sn = int(''.join([c for c in participant if c.isdigit()]))

        print(f'loading {participant}...')

        npz = np.load(os.path.join(gl.baseDir, experiment, participant, 'emg', f'{experiment}_{sn}_distances.npz'))
        dist.append(npz['data_array'])
        descr = json.loads(npz['descriptor'].item())
        win_size = descr['win_size']
        fsample = 2148

    dist = np.array(dist)

    timeAx = (np.linspace(-1 + win_size / (fsample * 2), 2 - win_size / (fsample * 2), dist.shape[-1]) -
              latency[['ring', 'index']].mean(axis=1).to_numpy())

    fig, axs = plt.subplots()

    colors = ['purple', 'green', 'darkorange']
    labels = ['finger', 'cue', 'interaction']
    sig_lev = [1.7, 1.75, 1.8]

    t_res = ttest_1samp(dist, 0, axis=0, alternative='greater')
    pval = t_res.pvalue < .05  # corrected with fdr remain just below 0.06
    pval_diff = np.diff(np.concatenate([pval.astype(int), np.zeros((3, 1))], axis=1))

    for i in range(dist.shape[1]):
        y = dist.mean(axis=0)[i]
        yerr = dist.std(axis=0)[i] / np.sqrt(dist.shape[0])
        axs.plot(timeAx, y, label=labels[i], color=colors[i])
        axs.fill_between(timeAx, y - yerr, y + yerr, color=colors[i], alpha=.3, lw=0)

        start = np.where(pval_diff[i] == 1)[0] / fsample - 1 - latency[['index', 'ring']].mean(axis=1).to_numpy()
        stop = np.where(pval_diff[i] == -1)[0] / fsample - 1 - latency[['index', 'ring']].mean(axis=1).to_numpy()
        interval = stop - start
        start = start[interval > .01]
        stop = stop[interval > .01]
        for j, k in zip(start, stop):
            axs.hlines(sig_lev[i], j, k,  color=colors[i], lw=3)

    axs.set_xlabel('time relative to stimulation (s)', fontsize=16)
    axs.set_ylabel('cross-validated distance (a.u.)', fontsize=16)

    axs.axvline(0, color='k', ls='-', lw=.8)
    axs.axvline(.025, color='k', ls=':', lw=.8)
    axs.axvline(.05, color='k', ls='--', lw=.8)
    axs.axvline(.1, color='k', ls='-.', lw=.8)
    axs.axhline(0, color='k', ls='-', lw=.8)

    axs.set_xlim([-.05, .5])
    # axs.set_ylim([0, 1.25])
    # axs.set_yscale('symlog', linthresh=.05)

    axs.legend(ncols=1, loc='upper right', ncol=1, fontsize=12)

    axs.set_title('cross-validated distance over time', fontsize=16)

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'dist.timec.svg'))

    plt.show()
