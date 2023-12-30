import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import smp0.experiment as exp

# from smp0.utils import detect_response_latency

matplotlib.use('MacOSX')


def plot_response_by_probability(data, datatype='emg'):
    tAx = exp.timeS[datatype]
    # tAx_clamped = exp.timeS['mov']
    #
    # latency_clamped = np.array((detect_response_latency(clamped[0, 1], threshold=.025, fsample=exp.fsample_mov),
    #                             detect_response_latency(clamped[1, 3], threshold=.025, fsample=exp.fsample_mov))) \
    #                   + tAx_clamped[0]

    # if datatype == 'mov':
    #     data = data - clamped

    # plot
    fig, axs = plt.subplots(len(exp.channels[datatype]), 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))

    # create colors and sort
    cmap = mcolors.LinearSegmentedColormap.from_list('red_to_blue', ['red', 'blue'], N=100)
    norm = plt.Normalize(0, 5 - 1)
    labels = list(exp.conditions['cues'].keys())
    sorted_indices = sorted(range(len([int(p.strip('%')) for p in labels])),
                            key=lambda i: [int(p.strip('%')) for p in labels][i])
    positions = [sorted_indices.index(i) for i in range(len(labels))]
    colors = [cmap(norm(i)) for i in range(len(labels))]
    sorted_colors = [colors[index] for index in positions]

    print('plotting...')

    for f, stimFinger in enumerate(data.keys()):
        for p, cue in enumerate(data[stimFinger].keys()):
            for c, ch in enumerate(data[stimFinger][cue].keys()):
                if np.array(data[stimFinger][cue][ch]).size != 0:
                    y = np.array(data[stimFinger][cue][ch]).mean(axis=0)
                    axs[c, f].plot(tAx, y, color=sorted_colors[p], label=labels[p])

                # axs[f, i].axvline(latency_clamped[i], ls='-', color='k', lw=1)
                # axs[f, i].axvline(latency_clamped[i] + .05, ls='-.', color='grey', lw=1)
                # axs[f, i].axvline(latency_clamped[i] + .1, ls=':', color='grey', lw=1)
            # axin = axs[ch, i].twinx()
            # if datatype == 'mov':
            #     axin.plot(tAx_clamped, clamped[i, ch], lw=1, ls='--', color='k')
            #     axin.set_ylim(axs[ch, i].get_ylim())
            #     fig.supylabel('Force (N)')
            # elif datatype == 'emg':
            #     fig.supylabel('EMG (mV)')
            #     #     # axin0.plot(tAx_clamped, clamped[0, 1], lw=1, ls='--', color='k')
            # #     # axin1.plot(tAx_clamped, clamped[1, 3], lw=1, ls='--', color='k')
            # #     axin0.set_ylim([0, 10])
            # #     axin1.set_ylim([0, 10])
    axs[0, 0].set_xlim([-.1, .5])
    axs[0, 0].set_ylim([0, None])
    axs[0, 0].legend(ncol=5, fontsize=6)
    # axs[0, 1].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    # # fig.tight_layout()

    fig.supxlabel('time (s)')
    # fig.suptitle(f"subj{participant_id}")
    plt.show()

# def plot_euclidean_distance_over_time(experiment, participant_id):
#     MyEmg = Emg(experiment, participant_id)
#     dist, dist_win, labels = MyEmg.euclidean_distance_probability()
#
#     # time axis
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
# fig2, axs2 = plt.subplots(constrained_layout=True, figsize=(8, 8))
#
# h = axs2.matshow(dist_win)
# axs2.set_yticks(np.linspace(0, len(labels) - 1, len(labels)))
# axs2.set_xticks(np.linspace(0, len(labels) - 1, len(labels)))
# axs2.set_xticklabels(labels, rotation=90)
# axs2.set_yticklabels(labels)
#
# fig2.colorbar(h)
#
# plt.show()
