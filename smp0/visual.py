import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smp0.info import timeS

# from load_data import load_dat

matplotlib.use('MacOSX')


def plot_response_by_probability(data, clamped, datatype=None):
    tAx = timeS[datatype]
    tAx_clamped = timeS['mov']

    # plot
    fig, axs = plt.subplots(data.shape[2], 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 7))

    # create colors
    shades = np.linspace(100, 200, data.shape[1]) / 255

    print('plotting...')

    for ch in range(data.shape[2]):
        for c in range(data.shape[1]):
            axs[ch, 0].plot(tAx, data[0, c, ch], color=[np.flip(shades)[c], np.flip(shades)[c], np.flip(shades)[c]])
            axs[ch, 1].plot(tAx, data[1, c, ch], color=[np.flip(shades)[c], np.flip(shades)[c], np.flip(shades)[c]])
        axin0 = axs[ch, 0].twinx()
        axin1 = axs[ch, 1].twinx()
        if datatype == 'mov':
            axin0.plot(tAx_clamped, clamped[0, ch], lw=1, ls='--', color='k')
            axin1.plot(tAx_clamped, clamped[1, ch], lw=1, ls='--', color='k')
            axin0.set_ylim(axs[ch, 0].get_ylim())
            axin1.set_ylim(axs[ch, 1].get_ylim())
        elif datatype == 'emg':
            axin0.plot(tAx_clamped, clamped[0, 1], lw=1, ls='--', color='k')
            axin1.plot(tAx_clamped, clamped[1, 3], lw=1, ls='--', color='k')
            axin0.set_ylim([0, 10])
            axin1.set_ylim([0, 10])
    axs[0, 0].set_xlim([-.1, .5])
    # axs[0, 0].set_ylim([0, None])
    axs[0, 0].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    # axs[0, 1].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    # # fig.tight_layout()
    # fig.supylabel('Force (N)')
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
