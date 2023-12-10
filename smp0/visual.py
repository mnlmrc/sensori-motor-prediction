import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smp0.emg import Emg
# from load_data import load_dat

matplotlib.use('MacOSX')


def plot_response_emg_by_finger(experiment, participant_id):

    MyEmg = Emg(experiment, participant_id)

    # sort by stimulated finger
    emg_index = MyEmg.sort_by_stimulated_finger(MyEmg.emg, "index")
    emg_ring = MyEmg.sort_by_stimulated_finger(MyEmg.emg, "ring")

    # time axis
    time = MyEmg.timeS

    # plot
    fig, axs = plt.subplots(len(MyEmg.muscle_names),
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 9))

    muscle_names = MyEmg.muscle_names

    meanIndex = emg_index.mean(axis=0)
    meanRing = emg_ring.mean(axis=0)

    for m, muscle in enumerate(muscle_names):
        axs[m].plot(time, meanIndex[m], color='r')
        axs[m].set_title(muscle, fontsize=6)
        axs[m].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m].axvline(x=.1, ls='--', color='k', lw=.8)
        axs[m].plot(time, meanRing[m], color='b')
        axs[m].set_title(muscle, fontsize=6)
        axs[m].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m].axvline(x=.1, ls='--', color='k', lw=.8)

    axs[0].set_xlim([-.1, .5])
    axs[0].set_ylim([0, None])
    axs[0].legend(['index', 'ring'], ncol=2, fontsize=6)
    # fig.tight_layout()
    fig.supylabel('EMG (mV)')
    fig.supxlabel('time (s)')
    fig.suptitle(f"subj{participant_id}")
    plt.show()

    return emg_index, emg_ring, time


def plot_synergies(experiment, participant_id):

    MyEmg = Emg(experiment, participant_id)
    # W, H, n, r_squared = MyEmg.nnmf_over_time()

    muscle_names = MyEmg.muscle_names
    n = MyEmg.syn['n_components']
    W = MyEmg.W
    r_squared = MyEmg.syn['r_squared']

    print(f"N components: {n} - $R^2$: {r_squared}")

    # plot
    fig, axs = plt.subplots(W.shape[-1],
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 9))

    for c, syn in enumerate(W.T):
        axs[c].bar(muscle_names, syn)


def plot_response_synergy_by_finger(experiment, participant_id):

    MyEmg = Emg('smp0', '100')

    # sort by stimulated finger
    syn_index = MyEmg.sort_by_stimulated_finger(MyEmg.H, "index")
    syn_ring = MyEmg.sort_by_stimulated_finger(MyEmg.H, "ring")
#
#     # time axis
#     time = MyEmg.timeS
#
#     # plot
#     fig, axs = plt.subplots(len(MyEmg.muscle_names),
#                             sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 9))
#
#     muscle_names = MyEmg.muscle_names
#
#     meanIndex = emg_index.mean(axis=0)
#     meanRing = emg_ring.mean(axis=0)
#     for m, muscle in enumerate(muscle_names):
#         axs[m].plot(time, meanIndex[m], color='r')
#         axs[m].set_title(muscle, fontsize=6)
#         axs[m].axvline(x=0, ls='-', color='k', lw=.8)
#         axs[m].axvline(x=.05, ls=':', color='k', lw=.8)
#         axs[m].axvline(x=.1, ls='--', color='k', lw=.8)
#         axs[m].plot(time, meanRing[m], color='b')
#         axs[m].set_title(muscle, fontsize=6)
#         axs[m].axvline(x=0, ls='-', color='k', lw=.8)
#         axs[m].axvline(x=.05, ls=':', color='k', lw=.8)
#         axs[m].axvline(x=.1, ls='--', color='k', lw=.8)
#
#     axs[0].set_xlim([-.1, .5])
#     axs[0].set_ylim([0, None])
#     axs[0].legend(['index', 'ring'], ncol=2, fontsize=6)
#     # fig.tight_layout()
#     fig.supylabel('EMG (mV)')
#     fig.supxlabel('time (s)')
#     fig.suptitle(f"subj{participant_id}")
#     plt.show()
#
#     return emg_index, emg_ring, time


def plot_response_emg_by_probability(experiment, participant_id):

    MyEmg = Emg(experiment, participant_id)

    # sort by cue and stimulated finger
    emg_index_25 = MyEmg.sort_by_stimulated_probability(MyEmg.emg, finger='index', cue="index 25% - ring 75%")
    emg_index_50 = MyEmg.sort_by_stimulated_probability(MyEmg.emg, finger='index', cue="index 50% - ring 50%")
    emg_index_75 = MyEmg.sort_by_stimulated_probability(MyEmg.emg, finger='index', cue="index 75% - ring 25%")
    emg_index_100 = MyEmg.sort_by_stimulated_probability(MyEmg.emg, finger='index', cue="index 100% - ring 0%")
    emg_ring_25 = MyEmg.sort_by_stimulated_probability(MyEmg.emg, finger="ring", cue="index 75% - ring 25%")
    emg_ring_50 = MyEmg.sort_by_stimulated_probability(MyEmg.emg, finger="ring", cue="index 50% - ring 50%")
    emg_ring_75 = MyEmg.sort_by_stimulated_probability(MyEmg.emg, finger="ring", cue="index 25% - ring 75%")
    emg_ring_100 = MyEmg.sort_by_stimulated_probability(MyEmg.emg, finger="ring", cue="index 0% - ring 100%")

    # time axis
    time = MyEmg.timeS

    # plot
    fig, axs = plt.subplots(len(MyEmg.muscle_names), 2,
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

    for m, muscle in enumerate(MyEmg.muscle_names):
        axs[m, 0].plot(time, emg_index_25.mean(axis=0)[m], color=red[0])
        axs[m, 0].plot(time, emg_index_50.mean(axis=0)[m], color=red[1])
        axs[m, 0].plot(time, emg_index_75.mean(axis=0)[m], color=red[2])
        axs[m, 0].plot(time, emg_index_100.mean(axis=0)[m], color=red[3])
        axs[m, 0].set_title(muscle, fontsize=6)
        axs[m, 0].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m, 0].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m, 0].axvline(x=.1, ls='--', color='k', lw=.8)

        axs[m, 1].plot(time, emg_ring_25.mean(axis=0)[m], color=blue[0])
        axs[m, 1].plot(time, emg_ring_50.mean(axis=0)[m], color=blue[1])
        axs[m, 1].plot(time, emg_ring_75.mean(axis=0)[m], color=blue[2])
        axs[m, 1].plot(time, emg_ring_100.mean(axis=0)[m], color=blue[3])
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
