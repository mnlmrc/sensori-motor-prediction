import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from smp0.experiment import Param
from smp0.utils import detect_response_latency

# from smp0.utils import detect_response_latency

matplotlib.use('MacOSX')


def plot_stim_aligned(M, err, clamped, latency_clamped, channels=None, datatype=None):

    n_cond = M.shape[1]

    tAx = Param(datatype).timeAx() - latency_clamped[0], Param(datatype).timeAx() - latency_clamped[1]
    tAx_clamped = Param('mov').timeAx() - latency_clamped[0], Param('mov').timeAx() - latency_clamped[1]

    # plot
    fig, axs = plt.subplots(len(channels), 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))

    # create colors and sort
    cmap = mcolors.LinearSegmentedColormap.from_list('red_to_blue', ['red', 'blue'], N=100)
    norm = plt.Normalize(0, 5 - 1)
    labels = ['0%', '25%', '50%', '75%', '100%']
    # sorted_indices = sorted(range(len([int(p.strip('%')) for p in labels])),
    #                         key=lambda i: [int(p.strip('%')) for p in labels][i])
    # positions = [sorted_indices.index(i) for i in range(len(labels))]
    colors = [cmap(norm(i)) for i in range(len(labels))]
    # sorted_colors = [colors[index] for index in positions]
    # sorted_labels = [labels[index] for index in sorted_indices]

    for ch, channel in enumerate(channels):
        for cue in range(4):
            axs[ch, 0].plot(tAx[0], M[channel][cue], color=colors[cue+1])
            axs[ch, 0].fill_between(tAx[0], M[channel][cue] + err[channel][cue],
                                    M[channel][cue] - err[channel][cue], color=colors[cue+1], alpha=.2, lw=0)
            axs[ch, 1].plot(tAx[1], M[channel][cue + 4], color=colors[cue])
            axs[ch, 1].fill_between(tAx[1], M[channel][cue + 4] + err[channel][cue + 4],
                                    M[channel][cue + 4] - err[channel][cue + 4], color=colors[cue], alpha=.2,
                                    lw=0)

        axs[ch, 0].set_title(channel, fontsize=7)
        axs[ch, 1].set_title(channel, fontsize=7)
        axs[ch, 0].axvline(0, ls='-', color='k', lw=1)
        axs[ch, 0].axvline(.05, ls='-.', color='grey', lw=1)
        axs[ch, 0].axvline(.1, ls=':', color='grey', lw=1)
        axs[ch, 1].axvline(0, ls='-', color='k', lw=1)
        axs[ch, 1].axvline(.05, ls='-.', color='grey', lw=1)
        axs[ch, 1].axvline(.1, ls=':', color='grey', lw=1)

    axs[0, 0].set_xlim([-.1, .5])
    axs[0, 0].set_ylim([0, None])

    if datatype == 'mov':
        for ch in range(len(channels)):
            axs[ch, 0].plot(tAx_clamped[0], clamped[0, ch], lw=1, ls='--', color='k')
            axs[ch, 1].plot(tAx_clamped[1], clamped[1, ch], lw=1, ls='--', color='k')

        fig.supylabel('Force (N)')

    elif datatype == 'emg':
        fig.supylabel('EMG (mV)')

    # for lab in sorted_labels:
    for color, label in zip(colors, labels):
        axs[0, 0].plot(np.nan, label=label, color=color)
    fig.legend(ncol=3, fontsize=6, loc='upper center')

    fig.supxlabel('time (s)')
    plt.show()


def plot_binned(M, err, wins, channels=None, datatype=None):
    bAx = np.linspace(1, len(M[channels[0]]) / 2, int(len(M[channels[0]]) / 2))

    fig, axs = plt.subplots(len(channels), 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))

    # create colors and sort
    cmap = mcolors.LinearSegmentedColormap.from_list('red_to_blue', ['red', 'blue'], N=100)
    norm = plt.Normalize(0, 5 - 1)
    labels = list(Exp.condition_codes['cues'].keys())
    colors = [cmap(norm(i)) for i in range(len(labels))]

    offset = .1
    for ch, channel in enumerate(channels):
        for cue in range(4):
            x_values_with_offset = bAx + (cue - 2) * offset
            axs[ch, 0].bar(x_values_with_offset, M[channel][cue],
                           width=offset, color=colors[cue+1], yerr=err[channel][cue])
            axs[ch, 1].bar(x_values_with_offset, M[channel][cue+4],
                           width=offset, color=colors[cue], yerr=err[channel][cue])

        axs[ch, 0].set_title(channel, fontsize=7)
        axs[ch, 1].set_title(channel, fontsize=7)

    axs[-1, 0].set_xticks(np.linspace(1, 4, 4))
    axs[-1, 0].set_xticklabels([f"{win[0]}-{win[1]}" for win in wins])

    if datatype == 'mov':
        fig.supylabel('Force (N)')

    elif datatype == 'emg':
        fig.supylabel('EMG (mV)')

    for color, label in zip(colors, labels):
        axs[0, 0].bar(np.nan, np.nan, label=label, color=color)
    fig.legend(ncol=3, fontsize=6, loc='upper center')

    fig.supxlabel('time window (s)')
    plt.show()

# def plot_response_binned(data, clamped, wins, datatype=None):
#     latency_clamped = np.array((detect_response_latency(clamped[0, 1],
#                                                         threshold=.025, fsample=exp.fsample_mov),
#                                 detect_response_latency(clamped[1, 3],
#                                                         threshold=.025, fsample=exp.fsample_mov))) - exp.prestim
#     tAx = exp.timeS[datatype] - latency_clamped[0], exp.timeS[datatype] - latency_clamped[0]
#     tAx_clamped = exp.timeS['mov'] - latency_clamped[0], exp.timeS['mov'] - latency_clamped[0]
#
#     fsample = exp.fsample[datatype]
#     idx = [(win[0] * fsample, win[1] * fsample) for win in wins]
#
#     # plot
#     fig, axs = plt.subplots(3, 2,
#                             sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))
#
#     for f, stimFinger in enumerate(data.keys()):
#         for p, cue in enumerate(data[stimFinger].keys()):
#             for c, ch in enumerate(data[stimFinger][cue].keys()):
#                 if np.array(data[stimFinger][cue][ch]).size != 0:
#                     y = np.array(data[stimFinger][cue][ch]).mean(axis=0)
#                     ywin = [y[i[0]:i[1]].mean() for i in idx]
#                     for win in wins:
