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
                                                   'subj110'], help='')

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

        npz = np.load(os.path.join(gl.baseDir, experiment, participant, 'emg', f'{experiment}_{sn}_distances.npz'))
        dist.append(npz['data_array'])
        descr = json.loads(npz['descriptor'].item())
        win_size = descr['win_size']


    dist = np.array(dist)

    timeAx = (np.linspace(-1 + win_size / (fsample * 2), 2 - win_size / (fsample * 2), dist.shape[-1]) -
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

        # start = np.where(pval_diff[i] == 1)[0] / fsample - 1 - latency[['index', 'ring']].mean(axis=1).to_numpy()
        # stop = np.where(pval_diff[i] == -1)[0] / fsample - 1 - latency[['index', 'ring']].mean(axis=1).to_numpy()
        # interval = stop - start
        # start = start[interval > .01]
        # stop = stop[interval > .01]
        # for j, k in zip(start, stop):
        #     axs.hlines(sig_lev[i], j, k,  color=colors[i], lw=3)

    axs.set_xlabel('time relative to stimulation (s)', fontsize=12)
    axs.set_ylabel('cross-validated multivariate distance (a.u.)', fontsize=12)

    axs.axvline(0, color='k', ls='-', lw=.8)
    axs.axvline(.025, color='k', ls=':', lw=.8)
    axs.axvline(.05, color='k', ls='--', lw=.8)
    axs.axvline(.1, color='k', ls='-.', lw=.8)
    axs.axhline(0, color='k', ls='-', lw=.8)

    axs.set_xlim([-.05, .2])
    axs.set_ylim([0, 1.2])
    # axs.set_yscale('symlog', linthresh=.05)

    fig.legend(ncols=1, loc='upper right', ncol=2, fontsize=12)

    # axs.set_title('cross-validated distance over time', fontsize=16)

    ori = [-.0375, .025, .0625, .125]
    tit = ['Pre', 'SLR', 'LLR', 'Vol']
    for w, o in enumerate(ori):
        axin = axs.inset_axes([o, 0.6, 0.025, .5], transform=axs.transData, zorder=-1)
        df = pd.DataFrame(columns=['finger', 'cue'])
        df.finger = dist[:, 0, win[w][0]:win[w][1]].mean(axis=-1)
        df.cue = dist[:, 1, win[w][0]:win[w][1]].mean(axis=-1)

        sns.boxplot(df, ax=axin, palette={'finger': 'purple', 'cue': 'darkorange'})
        axin.set_ylim([-.035, 3])
        axin.set_yscale('symlog', linthresh=.05)
        axin.set_title(tit[w])

        if w<3:
            axin.spines[['bottom', 'top', 'right','left' ]].set_visible(False)
            axin.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
            axin.set_xticks([])
        elif w==3:
            axin.spines[['bottom', 'top', 'left',]].set_visible(False)
            axin.tick_params(axis='both', which='both', length=2, labelbottom=False, labelright=True, labelleft=False)
            axin.set_xticks([])
            axin.set_ylabel('distance (a.u.)')
            axin.yaxis.set_ticks_position('right')  # Ensure y-axis ticks are on the left
            axin.yaxis.set_label_position('right')  # Ensure y-axis label is on the left



        t_stat_col1, p_value_col1 = ttest_1samp(df['finger'], 0,alternative='greater')
        t_stat_col2, p_value_col2 = ttest_1samp(df['cue'], 0,alternative='greater')

        print(f'{tit[w]}, finger - t-statistic: {t_stat_col1}, p-value: {p_value_col1}')
        print(f'{tit[w]}, cue - t-statistic: {t_stat_col2}, p-value: {p_value_col2}')



    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'dist.timec.svg'))

    plt.show()
