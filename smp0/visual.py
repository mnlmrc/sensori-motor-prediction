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

    meanIndex = emg_index.mean(axis=1)
    meanRing = emg_ring.mean(axis=1)
    for m, muscle in enumerate(muscle_names):
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
    df_emg_clean = df_emg.drop(['BN', 'TN', 'subNum', 'chordID', 'stimFinger', 'time'], axis=1)

    time = df_emg['time'][0]

    (emg_index_25, emg_index_50, emg_index_75, emg_index_100,
     emg_ring_25, emg_ring_50, emg_ring_75, emg_ring_100) = (
        [], [], [], [], [], [], [], [])
    for index, trial in df_emg.iterrows():
        if trial['stimFinger'] == 91999:
            match trial['chordID']:
                case 12:
                    trial_clean = df_emg_clean.iloc[index]
                    X = np.array([trial_clean[i] for i in trial_clean.index])
                    emg_index_25.append(X)
                case 44:
                    trial_clean = df_emg_clean.iloc[index]
                    X = np.array([trial_clean[i] for i in trial_clean.index])
                    emg_index_50.append(X)
                case 21:
                    trial_clean = df_emg_clean.iloc[index]
                    X = np.array([trial_clean[i] for i in trial_clean.index])
                    emg_index_75.append(X)
                case 39:
                    trial_clean = df_emg_clean.iloc[index]
                    X = np.array([trial_clean[i] for i in trial_clean.index])
                    emg_index_100.append(X)
        elif trial['stimFinger'] == 99919:
            match trial['chordID']:
                case 12:
                    trial_clean = df_emg_clean.iloc[index]
                    X = np.array([trial_clean[i] for i in trial_clean.index])
                    emg_ring_25.append(X)
                case 44:
                    trial_clean = df_emg_clean.iloc[index]
                    X = np.array([trial_clean[i] for i in trial_clean.index])
                    emg_ring_50.append(X)
                case 21:
                    trial_clean = df_emg_clean.iloc[index]
                    X = np.array([trial_clean[i] for i in trial_clean.index])
                    emg_ring_75.append(X)
                case 93:
                    trial_clean = df_emg_clean.iloc[index]
                    X = np.array([trial_clean[i] for i in trial_clean.index])
                    emg_ring_100.append(X)

    emg_index_25 = np.array(emg_index_25).swapaxes(0, 1)
    emg_index_50 = np.array(emg_index_50).swapaxes(0, 1)
    emg_index_75 = np.array(emg_index_75).swapaxes(0, 1)
    emg_index_100 = np.array(emg_index_100).swapaxes(0, 1)
    emg_ring_25 = np.array(emg_ring_25).swapaxes(0, 1)
    emg_ring_50 = np.array(emg_ring_50).swapaxes(0, 1)
    emg_ring_75 = np.array(emg_ring_75).swapaxes(0, 1)
    emg_ring_100 = np.array(emg_ring_100).swapaxes(0, 1)

    muscle_names = df_emg_clean.columns.to_list()

    fig, axs = plt.subplots(len(muscle_names), 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 7))

    base = 255
    red = [
        (1, base / 255, base / 255),
        (1, (base - 60) / 255, (base - 60) / 255),
        (1, (base - 120) / 255, (base - 120) / 255),
        (1, (base - 180) / 255, (base - 180) / 255)]

    meanIndex25 = emg_index_25.mean(axis=1)
    meanIndex50 = emg_index_50.mean(axis=1)
    meanIndex75 = emg_index_75.mean(axis=1)
    meanIndex100 = emg_index_100.mean(axis=1)
    meanRing25 = emg_ring_25.mean(axis=1)
    meanRing50 = emg_ring_50.mean(axis=1)
    meanRing75 = emg_ring_75.mean(axis=1)
    meanRing100 = emg_ring_100.mean(axis=1)
    for m, muscle in enumerate(muscle_names):
        axs[m, 0].plot(time, meanIndex25[m], color=red[0])
        axs[m, 0].plot(time, meanIndex50[m], color=red[1])
        axs[m, 0].plot(time, meanIndex75[m], color=red[2])
        axs[m, 0].plot(time, meanIndex100[m], color=red[3])
        axs[m, 0].set_title(muscle, fontsize=6)
        axs[m, 0].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m, 0].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m, 0].axvline(x=.1, ls='--', color='k', lw=.8)
        # axs[m, 1].plot(time, meanRing[m], color='b')
        # axs[m, 1].set_title(muscle, fontsize=6)
        # axs[m, 1].axvline(x=0, ls='-', color='k', lw=.8)
        # axs[m, 1].axvline(x=.05, ls=':', color='k', lw=.8)
        # axs[m, 1].axvline(x=.1, ls='--', color='k', lw=.8)

    axs[0, 0].set_xlim([-.1, .5])
    axs[0, 0].set_ylim([0, None])
    # fig.tight_layout()
    fig.supylabel('EMG (mV)')
    fig.supxlabel('time (s)')
    fig.suptitle(f"subj{participant_id}")
    plt.show()



path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path

# emg_index, emg_ring, time = plot_response_emg_by_finger(experiment='smp0', participant_id='100')

X = plot_response_emg_by_probability(experiment='smp0', participant_id='100')
