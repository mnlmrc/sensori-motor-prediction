import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('MacOSX')


def plot_response_emg_by_finger(experiment, participant_id):
    fname = f"{experiment}_{participant_id}.emg"
    filepath = os.path.join(path, experiment, f"subj{participant_id}", 'emg', fname)
    df_emg = pd.read_pickle(filepath)

    time = df_emg['time'][0]

    df_emg_index = df_emg[df_emg['stimFinger'] == 91999].drop(['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
                                                              axis=1)
    df_emg_ring = df_emg[df_emg['stimFinger'] == 99919].drop(['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
                                                             axis=1)

    muscle_names = df_emg_index.columns.to_list()
    ntrials = len(df_emg_index)

    emg_index = np.zeros((len(muscle_names), ntrials, len(time)))
    emg_ring = np.zeros((len(muscle_names), ntrials, len(time)))
    for m, muscle in enumerate(muscle_names):
        for ntrial in range(ntrials):
            emg_index[m, ntrial] = df_emg_index[muscle].iloc[ntrial]
            emg_ring[m, ntrial] = df_emg_ring[muscle].iloc[ntrial]

    fig, axs = plt.subplots(len(muscle_names), 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 7))

    for m, muscle in enumerate(muscle_names):
        meanIndex = emg_index.mean(axis=1)
        meanRing = emg_ring.mean(axis=1)
        axs[m, 0].plot(time, meanIndex[m], color='r')
        axs[m, 0].set_title(muscle, fontsize=6)
        axs[m, 0].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m, 0].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m, 0].axvline(x=.1, ls='--', color='k', lw=.8)
        axs[m, 1].plot(time, meanRing[m], color='b')
        axs[m, 1].set_title(muscle, fontsize=6)
        axs[m, 1].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m, 1].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m, 1].axvline(x=.1, ls='--', color='k', lw=.8)

    axs[0, 0].set_xlim([-.1, .5])
    axs[0, 0].set_ylim([0, None])
    # fig.tight_layout()
    fig.supylabel('EMG (mV)')
    fig.supxlabel('time (s)')
    fig.suptitle(f"subj{participant_id}")
    plt.show()

    return emg_index, emg_ring, time


def plot_response_emg_by_probability(experiment, participant_id):
    fname = f"{experiment}_{participant_id}.emg"
    filepath = os.path.join(path, experiment, f"subj{participant_id}", 'emg', fname)
    df_emg = pd.read_pickle(filepath)

    time = df_emg['time'][0]

    df_emg_index_25 = df_emg[(df_emg['stimFinger'] == 91999) & (df_emg['chordID'] == 12)].drop(['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
                                                              axis=1)
    df_emg_index_50 = df_emg[(df_emg['stimFinger'] == 91999) & (df_emg['chordID'] == 44)].drop(['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
                                                              axis=1)
    df_emg_index_75 = df_emg[(df_emg['stimFinger'] == 91999) & (df_emg['chordID'] == 21)].drop(['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
                                                              axis=1)
    df_emg_index_100 = df_emg[(df_emg['stimFinger'] == 91999) & (df_emg['chordID'] == 39)].drop(['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
                                                              axis=1)

    df_emg_ring_25 = df_emg[(df_emg['stimFinger'] == 99919) & (df_emg['chordID'] == 12)].drop(
        ['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
        axis=1)
    df_emg_ring_50 = df_emg[(df_emg['stimFinger'] == 99919) & (df_emg['chordID'] == 44)].drop(
        ['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
        axis=1)
    df_emg_ring_75 = df_emg[(df_emg['stimFinger'] == 99919) & (df_emg['chordID'] == 21)].drop(
        ['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
        axis=1)
    df_emg_ring_100 = df_emg[(df_emg['stimFinger'] == 99919) & (df_emg['chordID'] == 93)].drop(
        ['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'],
        axis=1)

    emg_index_25 = []
    emg_index_50 = []
    emg_index_75 = []
    emg_index_100 = []
    emg_ring_25 = []
    emg_ring_50 = []
    emg_ring_75 = []
    emg_ring_100 = []

    for trial in df_emg:
        if trial['stimFinger'] == 91999:
            match trial['chordID']:
                case '39':
                    



    muscle_names = df_emg_index_25.columns.to_list()
    ntrials = len(df_emg_index_25)


    for m, muscle in enumerate(muscle_names):
        for ntrial in range(ntrials):
            emg_index_25[m, ntrial] = df_emg_index_25[muscle].iloc[ntrial]
            emg_index_50[m, ntrial] = df_emg_index_50[muscle].iloc[ntrial]
            emg_index_75[m, ntrial] = df_emg_index_75[muscle].iloc[ntrial]
            emg_index_100[m, ntrial] = df_emg_index_100[muscle].iloc[ntrial]
            emg_ring_25[m, ntrial] = df_emg_ring_25[muscle].iloc[ntrial]
            emg_ring_50[m, ntrial] = df_emg_ring_50[muscle].iloc[ntrial]
            emg_ring_75[m, ntrial] = df_emg_ring_75[muscle].iloc[ntrial]
            emg_ring_100[m, ntrial] = df_emg_ring_100[muscle].iloc[ntrial]


path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path

# emg_index, emg_ring, time = plot_response_emg_by_finger(experiment='smp0', participant_id='100')

plot_response_emg_by_probability(experiment='smp0', participant_id='100')
