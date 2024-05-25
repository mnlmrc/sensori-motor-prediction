import argparse

import pandas as pd
import matplotlib.pyplot as plt
import os

import globals as gl
import seaborn as sns

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Plot RDM")
    parser.add_argument('--experiment', default='smp1', help='Experiment')
    parser.add_argument('--participants', nargs='+', default=['subj100',
                                                              'subj101',
                                                              'subj103'], help='Participants')
    parser.add_argument('--session',  default='training', help='Session')

    args = parser.parse_args()
    participants = args.participants
    experiment = args.experiment
    session = args.session

    path = os.path.join(gl.baseDir, experiment, session)

    forceDiff = pd.DataFrame()
    fig, axs = plt.subplots()
    window_size = 20
    for participant in participants:

        sn = int(''.join([c for c in participant if c.isdigit()]))

        dat = pd.read_csv(os.path.join(path, participant, f'{experiment}_{sn}.dat'), sep='\t')
        forceDiff = pd.concat([forceDiff, dat[['forceDiff', 'subNum']]])

        forceDiff_subj = dat['forceDiff'].abs().rolling(window=window_size).mean()
        axs.plot(forceDiff_subj, label=participant)

    axs.legend(loc='upper right')
    axs.set_ylabel('absolute force difference (N)')
    axs.set_xlabel('#trial')
    axs.set_title(f'Moving average (window={window_size}) of force '
                  f'difference\nbetween cued and non-cued finger in session {session}')

    fig, axs = plt.subplots()
    palette = sns.color_palette('tab10')
    sns.boxplot(forceDiff, y='forceDiff', x='subNum', ax=axs, palette=palette)
    axs.axhline(0, color='k', ls='--', lw=.8)
    axs.set_ylabel('force difference (N)')
    axs.set_title(f'Average force difference\nbetween cued and non-cued finger in session {session}')