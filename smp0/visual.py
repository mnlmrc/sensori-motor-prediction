import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smp0.participant import Emg, Force
from smp0.util import centered_moving_average, hotelling_t2_test_1_sample, filter_pval_series

# from load_data import load_dat

matplotlib.use('MacOSX')


def plot_response_emg_by_finger(experiment, participant_id):
    MyEmg = Emg(experiment, participant_id)

    # time axis
    time = MyEmg.timeS
    baseline = MyEmg.emg[..., np.where((time > -.1) & (time < 0))[0]].mean(axis=(0, -1))
    emg = MyEmg.emg
    fsample = MyEmg.fsample
    prestim = MyEmg.prestim

    T2, pval = np.zeros(len(time)), np.zeros(len(time))
    for t in range(len(time)):
        T2[t], pval[t] = hotelling_t2_test_1_sample(emg[..., t], baseline)

    _, start_timings = filter_pval_series(pval, int(.03 * fsample), threshold=0.05, fsample=fsample, prestim=prestim)

    # sort by stimulated finger
    emg_index = MyEmg.sort_by_stimulated_finger(emg, "index")
    emg_ring = MyEmg.sort_by_stimulated_finger(emg, "ring")

    # plot
    fig, axs = plt.subplots(len(MyEmg.muscle_names),
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 9))

    muscle_names = MyEmg.muscle_names

    meanIndex = emg_index.mean(axis=0)
    meanRing = emg_ring.mean(axis=0)

    for m, muscle in enumerate(muscle_names):
        axs[m].plot(time, meanIndex[m], color='r')
        axs[m].plot(time, meanRing[m], color='b')
        axs[m].set_title(muscle, fontsize=6)
        axs[m].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m].axvline(x=.1, ls='--', color='k', lw=.8)
        # axs[m].twinx().plot(time, pval_bool, color='k', lw=.8)
        axs[m].twinx().stem(start_timings, [1] * len(start_timings), linefmt='k-', markerfmt='ko', basefmt=" ",
                            label='Start Points')

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

    # def plot_response_synergy_by_finger(experiment, participant_id):
    #
    #     MyEmg = Emg('smp0', '100')

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


def plot_response_force_by_probability(experiment, participant_id):
    MyForce = Force(experiment, participant_id)
    MyForce.D_squared_over_time()

    channel_names = ["thumb", "index", "middle", "ring", "pinkie"]

    D_squared_mean = np.zeros((2, MyForce.D_squared.shape[-1]))
    D_squared_trad_mean = np.zeros((2, MyForce.D_squared.shape[-1]))
    for d in range(MyForce.D_squared.shape[-1]):

        row_indices, col_indices = np.triu_indices_from(MyForce.D_squared[0, ..., d], k=1)
        D_squared_mean[0, d] = MyForce.D_squared[0, row_indices, col_indices, d].mean()
        row_indices, col_indices = np.triu_indices_from(MyForce.D_squared[1, ..., d], k=1)
        D_squared_mean[1, d] = MyForce.D_squared[1, row_indices, col_indices, d].mean()

        # row_indices, col_indices = np.triu_indices_from(MyForce.D_squared_trad[0, ..., d], k=1)
        # D_squared_trad_mean[0, d] = MyForce.D_squared_trad[0, row_indices, col_indices, d].mean()
        # row_indices, col_indices = np.triu_indices_from(MyForce.D_squared_trad[1, ..., d], k=1)
        # D_squared_trad_mean[1, d] = MyForce.D_squared_trad[1, row_indices, col_indices, d].mean()

    # MyEmg.emg = centered_moving_average(MyEmg.emg, 11)

    # sort by cue and stimulated finger
    force_0 = MyForce.sort_by_stimulated_probability(finger='ring', cue="0%")
    force_25 = (MyForce.sort_by_stimulated_probability(finger='index', cue="25%"),
                MyForce.sort_by_stimulated_probability(finger='ring', cue="25%"))
    force_50 = (MyForce.sort_by_stimulated_probability(finger='index', cue="50%"),
                MyForce.sort_by_stimulated_probability(finger='ring', cue="50%"))
    force_75 = (MyForce.sort_by_stimulated_probability(finger='index', cue="75%"),
                MyForce.sort_by_stimulated_probability(finger='ring', cue="75%"))
    force_100 = (MyForce.sort_by_stimulated_probability(finger='index', cue="100%"))

    # time axis
    time = MyForce.timeS

    # plot
    fig, axs = plt.subplots(5, 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 7))

    # create colors
    base = 255
    blue = [
        ((base - 30) / 255, (base - 30) / 255, 1),  # Mid-tone blue
        ((base - 60) / 255, (base - 60) / 255, 1),
        ((base - 120) / 255, (base - 120) / 255, 1),
        ((base - 180) / 255, (base - 180) / 255, 1)
    ]
    
    lw = ((1, 2, 1, 1, 1), (1, 1, 1, 2, 1))

    for f in range(5):

        axs[f, 0].plot(time, force_25[0].mean(axis=0)[f], color=blue[0], lw=lw[0][f])
        axs[f, 0].plot(time, force_50[0].mean(axis=0)[f], color=blue[1], lw=lw[0][f])
        axs[f, 0].plot(time, force_75[0].mean(axis=0)[f], color=blue[2], lw=lw[0][f])
        axs[f, 0].plot(time, force_100.mean(axis=0)[f], color=blue[3], lw=lw[0][f])
        axin = axs[f, 0].twinx()
        axin.plot(time, D_squared_mean[0], color='r', lw=1)
        # axin.plot(time, D_squared_trad_mean[0], color='orange', lw=1)
        axs[f, 0].set_title(channel_names[f], fontsize=6)
        axs[f, 0].axvline(x=0, ls='-', color='k', lw=.8)
        axs[f, 0].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[f, 0].axvline(x=.1, ls='--', color='k', lw=.8)

        axs[f, 1].plot(time, force_0.mean(axis=0)[f], color=blue[0], lw=lw[1][f])
        axs[f, 1].plot(time, force_25[1].mean(axis=0)[f], color=blue[1], lw=lw[1][f])
        axs[f, 1].plot(time, force_50[1].mean(axis=0)[f], color=blue[2], lw=lw[1][f])
        axs[f, 1].plot(time, force_75[1].mean(axis=0)[f], color=blue[3], lw=lw[1][f])
        axin = axs[f, 1].twinx()
        axin.plot(time, D_squared_mean[1], color='r', lw=1)
        # axin.plot(time, D_squared_trad_mean[1], color='orange', lw=1)
        axs[f, 1].set_title(channel_names[f], fontsize=6)
        axs[f, 1].axvline(x=0, ls='-', color='k', lw=.8)
        axs[f, 1].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[f, 1].axvline(x=.1, ls='--', color='k', lw=.8)
        axs[f, 1].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[f, 1].axvline(x=.1, ls='--', color='k', lw=.8)

    axs[0, 0].set_xlim([-.1, .5])
    # axs[0, 0].set_ylim([0, None])
    axs[0, 0].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    axs[0, 1].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    # fig.tight_layout()
    fig.supylabel('Force (N)')
    fig.supxlabel('time (s)')
    fig.suptitle(f"subj{participant_id}")
    plt.show()


def plot_euclidean_distance_over_time(experiment, participant_id):
    MyEmg = Emg(experiment, participant_id)
    dist, dist_win, labels = MyEmg.euclidean_distance_probability()

    # time axis
    # time = MyEmg.timeS
    #
    # num_conditions = dist.shape[0]

    # # plot
    # fig1, axs1 = plt.subplots(len(labels), len(labels),
    #                         sharex=True, sharey=True,
    #                         constrained_layout=True, figsize=(8, 8))
    #
    # for i in range(num_conditions):
    #     for j in range(num_conditions):
    #         axs1[i, j].plot(time, dist[i, j], color='k')
    #
    #         if i == 0:
    #             axs1[i, j].set_title(labels[j], fontsize=10)
    #         if j == num_conditions - 1:
    #             axs1[i, j].set_ylabel(labels[i], fontsize=10)
    #             axs1[i, j].yaxis.set_label_position("right")
    #
    # axs1[0, 0].set_xlim([-.1, .5])
    #
    # plt.show()

    # plot
    fig2, axs2 = plt.subplots(constrained_layout=True, figsize=(8, 8))

    h = axs2.matshow(dist_win)
    axs2.set_yticks(np.linspace(0, len(labels) - 1, len(labels)))
    axs2.set_xticks(np.linspace(0, len(labels) - 1, len(labels)))
    axs2.set_xticklabels(labels, rotation=90)
    axs2.set_yticklabels(labels)

    fig2.colorbar(h)

    plt.show()
