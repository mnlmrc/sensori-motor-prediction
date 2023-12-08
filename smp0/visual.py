import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from emg import Emg
# from load_data import load_dat

matplotlib.use('MacOSX')


def plot_response_emg_by_finger(experiment, participant_id):

    myEmg = Emg(experiment, participant_id)

    # sort by stimulated finger
    emg_index = myEmg.sort_by_stimulated_finger("index")
    emg_ring = myEmg.sort_by_stimulated_finger("ring")

    # time axis
    time = myEmg.timeS

    # plot
    fig, axs = plt.subplots(len(myEmg.muscle_names), 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 7))

    muscle_names = myEmg.muscle_names

    meanIndex = emg_index.mean(axis=0)
    meanRing = emg_ring.mean(axis=0)
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

    # load segmented emg
    fname = f"{experiment}_{participant_id}.npy"
    filepath = os.path.join(path, experiment, f"subj{participant_id}", 'emg', fname)
    emg = np.load(filepath)

    # load .dat file
    D = load_dat(experiment, participant_id)

    # sort by cue and stimulated finger
    emg_index_25 = emg[D[(D['stimFinger'] == 91999) & (D['chordID'] == 12)].index, :-1]
    emg_index_50 = emg[D[(D['stimFinger'] == 91999) & (D['chordID'] == 44)].index, :-1]
    emg_index_75 = emg[D[(D['stimFinger'] == 91999) & (D['chordID'] == 21)].index, :-1]
    emg_index_100 = emg[D[(D['stimFinger'] == 91999) & (D['chordID'] == 39)].index, :-1]
    emg_ring_25 = emg[D[(D['stimFinger'] == 99919) & (D['chordID'] == 12)].index, :-1]
    emg_ring_50 = emg[D[(D['stimFinger'] == 99919) & (D['chordID'] == 44)].index, :-1]
    emg_ring_75 = emg[D[(D['stimFinger'] == 99919) & (D['chordID'] == 21)].index, :-1]
    emg_ring_100 = emg[D[(D['stimFinger'] == 99919) & (D['chordID'] == 93)].index, :-1]

    # time axis
    time = emg[0, -1]

    # plot
    fig, axs = plt.subplots(len(muscle_names), 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 7))

    # create colors
    base = 255
    red = [
        (1, (base - 30) / 255, (base - 30) / 255),
        (1, (base - 60) / 255, (base - 60) / 255),
        (1, (base - 120) / 255, (base - 120) / 255),
        (1, (base - 180) / 255, (base - 180) / 255)]
    blue = [
        ((base - 30) / 255, (base - 30) / 255, 1),
        ((base - 60) / 255, (base - 60) / 255, 1),
        ((base - 120) / 255, (base - 120) / 255, 1),
        ((base - 180) / 255, (base - 180) / 255, 1)
    ]

    meanIndex25 = emg_index_25.mean(axis=0)
    meanIndex50 = emg_index_50.mean(axis=0)
    meanIndex75 = emg_index_75.mean(axis=0)
    meanIndex100 = emg_index_100.mean(axis=0)
    meanRing25 = emg_ring_25.mean(axis=0)
    meanRing50 = emg_ring_50.mean(axis=0)
    meanRing75 = emg_ring_75.mean(axis=0)
    meanRing100 = emg_ring_100.mean(axis=0)
    for m, muscle in enumerate(muscle_names):
        axs[m, 0].plot(time, meanIndex25[m], color=red[0])
        axs[m, 0].plot(time, meanIndex50[m], color=red[1])
        axs[m, 0].plot(time, meanIndex75[m], color=red[2])
        axs[m, 0].plot(time, meanIndex100[m], color=red[3])
        axs[m, 0].set_title(muscle, fontsize=6)
        axs[m, 0].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m, 0].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m, 0].axvline(x=.1, ls='--', color='k', lw=.8)

        axs[m, 1].plot(time, meanRing25[m], color=blue[0])
        axs[m, 1].plot(time, meanRing50[m], color=blue[1])
        axs[m, 1].plot(time, meanRing75[m], color=blue[2])
        axs[m, 1].plot(time, meanRing100[m], color=blue[3])
        axs[m, 1].set_title(muscle, fontsize=6)
        axs[m, 1].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m, 1].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m, 1].axvline(x=.1, ls='--', color='k', lw=.8)
        axs[m, 1].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m, 1].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m, 1].axvline(x=.1, ls='--', color='k', lw=.8)

    axs[0, 0].set_xlim([-.1, .5])
    axs[0, 0].set_ylim([0, None])
    axs[0, 0].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    axs[0, 1].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    # fig.tight_layout()
    fig.supylabel('EMG (mV)')
    fig.supxlabel('time (s)')
    fig.suptitle(f"subj{participant_id}")
    plt.show()


# path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path
# muscle_names = ['thumb_flex', 'index_flex', 'middle_flex', 'ring_flex', 'pinkie_flex', 'thumb_ext', 'index_ext',
#                 'middle_ext', 'ring_ext', 'pinkie_ext']
# emg_index, emg_ring, time = plot_response_emg_by_finger(experiment='smp0', participant_id='100')

# X = plot_response_emg_by_finger(experiment='smp0', participant_id='103')
