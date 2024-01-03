import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import smp0.experiment as exp
from smp0.utils import detect_response_latency

# from smp0.utils import detect_response_latency

matplotlib.use('MacOSX')


def plot_response_continuous(data, clamped, datatype=None):

    latency_clamped = np.array((detect_response_latency(clamped[0, 1],
                                                        threshold=.025, fsample=exp.fsample_mov),
                                detect_response_latency(clamped[1, 3],
                                                        threshold=.025, fsample=exp.fsample_mov))) - exp.prestim
    tAx = exp.timeS[datatype] - latency_clamped[0], exp.timeS[datatype] - latency_clamped[0]
    tAx_clamped = exp.timeS['mov'] - latency_clamped[0], exp.timeS['mov'] - latency_clamped[0]
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
    sorted_labels = [labels[index] for index in sorted_indices]

    print('plotting...')

    for f, stimFinger in enumerate(data.keys()):
        for p, cue in enumerate(data[stimFinger].keys()):
            for c, ch in enumerate(data[stimFinger][cue].keys()):
                if np.array(data[stimFinger][cue][ch]).size != 0:
                    y = np.array(data[stimFinger][cue][ch]).mean(axis=0)
                    axs[c, f].plot(tAx[f], y, color=sorted_colors[p])
                    axs[c, f].set_title(ch, fontsize=6)

    axs[0, 0].set_xlim([-.1, .5])
    axs[0, 0].set_ylim([0, None])

    if datatype == 'mov':
        for f in range(len(data.keys())):
            for c in range(len(exp.channels[datatype])):
                axin = axs[c, f].twinx()
                axin.plot(tAx_clamped[f], clamped[f, c], lw=1, ls='--', color='k')
                axin.set_ylim(axs[c, f].get_ylim())

        fig.supylabel('Force (N)')

    elif datatype == 'emg':
        for f in range(len(data.keys())):
            for c in range(len(exp.channels[datatype])):
                axs[c, f].axvline(0, ls='-', color='k', lw=1)
                axs[c, f].axvline(.05, ls='-.', color='grey', lw=1)
                axs[c, f].axvline(.1, ls=':', color='grey', lw=1)
        #     # axin0.plot(tAx_clamped, clamped[0, 1], lw=1, ls='--', color='k')
    #     # axin1.plot(tAx_clamped, clamped[1, 3], lw=1, ls='--', color='k')
    #     axin0.set_ylim([0, 10])
    #     axin1.set_ylim([0, 10])
        fig.supylabel('EMG (mV)')

    # for lab in sorted_labels:
    for color, label in zip(colors, sorted_labels):
        axs[0, 0].plot(np.nan, label=label, color=color)
    fig.legend(ncol=3, fontsize=6, loc='upper center')
    # axs[0, 1].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    # # fig.tight_layout()

    fig.supxlabel('time (s)')
    # fig.suptitle(f"subj{participant_id}")
    plt.show()


def plot_response_binned(data, clamped, wins, datatype=None):
    latency_clamped = np.array((detect_response_latency(clamped[0, 1],
                                                        threshold=.025, fsample=exp.fsample_mov),
                                detect_response_latency(clamped[1, 3],
                                                        threshold=.025, fsample=exp.fsample_mov))) - exp.prestim
    tAx = exp.timeS[datatype] - latency_clamped[0], exp.timeS[datatype] - latency_clamped[0]
    tAx_clamped = exp.timeS['mov'] - latency_clamped[0], exp.timeS['mov'] - latency_clamped[0]

    fsample = exp.fsample[datatype]
    idx = [(win[0] * fsample, win[1] * fsample) for win in wins]

    # plot
    fig, axs = plt.subplots(3, 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))

    for f, stimFinger in enumerate(data.keys()):
        for p, cue in enumerate(data[stimFinger].keys()):
            for c, ch in enumerate(data[stimFinger][cue].keys()):
                if np.array(data[stimFinger][cue][ch]).size != 0:
                    y = np.array(data[stimFinger][cue][ch]).mean(axis=0)
                    ywin = [y[i[0]:i[1]].mean() for i in idx]
                    for win in wins:



