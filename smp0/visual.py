import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PcmPy import indicator

from smp0.experiment import Param
from smp0.utils import detect_response_latency, average_condition

# from smp0.utils import detect_response_latency

matplotlib.use('MacOSX')


def make_colors(n_labels, extreme_colors):
    cmap = mcolors.LinearSegmentedColormap.from_list(f"{extreme_colors[0]}_to_{extreme_colors[1]}",
                                                     [extreme_colors[0], extreme_colors[1]], N=100)
    norm = plt.Normalize(0, n_labels)
    colors = [cmap(norm(i)) for i in range(n_labels)]

    return colors


def plot_stim_aligned(M, err, labels=None, channels=None, datatype=None, ex_col=('red', 'blue')):
    # clamped, latency_clamped, channels=None, datatype=None):

    # n_cond = M.shape[1]

    tAx = Param(datatype).timeAx() - latency_clamped[0], Param(datatype).timeAx() - latency_clamped[1]
    tAx_clamped = Param('mov').timeAx() - latency_clamped[0], Param('mov').timeAx() - latency_clamped[1]

    # plot
    fig, axs = plt.subplots(len(channels), 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))

    # create colors and sort
    colors = make_colors(n_labels, ex_col)

    for ch, channel in enumerate(channels):
        for cue in range(4):
            axs[ch, 0].plot(tAx[0], M[channel][cue], color=colors[cue + 1])
            axs[ch, 0].fill_between(tAx[0], M[channel][cue] + err[channel][cue],
                                    M[channel][cue] - err[channel][cue], color=colors[cue + 1], alpha=.2, lw=0)
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


# def plot_binned(M, err, wins, channels=None, datatype=None):
#     bAx = np.linspace(1, len(M[channels[0]]) / 2, int(len(M[channels[0]]) / 2))
#
#     fig, axs = plt.subplots(len(channels), 2,
#                             sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))
#
#     # create colors and sort
#     cmap = mcolors.LinearSegmentedColormap.from_list('red_to_blue', ['red', 'blue'], N=100)
#     norm = plt.Normalize(0, 5 - 1)
#     labels = list(Exp.condition_codes['cues'].keys())
#     colors = [cmap(norm(i)) for i in range(len(labels))]
#
#     offset = .1
#     for ch, channel in enumerate(channels):
#         for cue in range(4):
#             x_values_with_offset = bAx + (cue - 2) * offset
#             axs[ch, 0].bar(x_values_with_offset, M[channel][cue],
#                            width=offset, color=colors[cue+1], yerr=err[channel][cue])
#             axs[ch, 1].bar(x_values_with_offset, M[channel][cue+4],
#                            width=offset, color=colors[cue], yerr=err[channel][cue])
#
#         axs[ch, 0].set_title(channel, fontsize=7)
#         axs[ch, 1].set_title(channel, fontsize=7)
#
#     axs[-1, 0].set_xticks(np.linspace(1, 4, 4))
#     axs[-1, 0].set_xticklabels([f"{win[0]}-{win[1]}" for win in wins])
#
#     if datatype == 'mov':
#         fig.supylabel('Force (N)')
#
#     elif datatype == 'emg':
#         fig.supylabel('EMG (mV)')
#
#     for color, label in zip(colors, labels):
#         axs[0, 0].bar(np.nan, np.nan, label=label, color=color)
#     fig.legend(ncol=3, fontsize=6, loc='upper center')
#
#     fig.supxlabel('time window (s)')
#     plt.show()


# def plot_binned(df, variables=None, categories=None):
#
#     df_agg = pd.pivot_table(df, values=variables, index=categories, aggfunc='mean').reset_index()
#
#
#     return df_agg

class Plotter3D:

    def __init__(self, data, channels=None, conditions=None, labels=None, extreme_colors=('red', 'blue'), figsize=(6.4, 4.8)):

        self.data = data
        self.channels = channels
        self.conditions = conditions
        self.labels = labels
        self.ecol = extreme_colors
        self.figsize = figsize
        # self._figsize_per_subplot = (figsize[0] / len(conditions), figsize[1] / len(channels))

    def _setup_subplots(self):
        n_conditions = len(self.conditions)
        n_channels = len(self.channels)
        self.fig, self.axs = plt.subplots(n_channels, n_conditions, figsize=self.figsize)
        if n_channels == 1 or n_conditions == 1:
            self.axs = np.array([self.axs])

    def _make_colors(self):

        n_labels = len(self.labels)
        cmap = mcolors.LinearSegmentedColormap.from_list(f"{self.ecol[0]}_to_{self.ecol[1]}",
                                                         [self.ecol[0], self.ecol[1]], N=100)
        norm = plt.Normalize(0, n_labels)
        colors = [cmap(norm(i)) for i in range(n_labels)]

        return colors

    def _av_across_participants(self):

        channels_dict = {ch: [] for ch in self.channels}
        N = len(self.data)
        for p_data in self.data:
            Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
            M = average_condition(p_data, Z)

            for ch in self.channels:
                if ch in p_data.channel_descriptors['channels']:
                    channel_index = p_data.channel_descriptors['channels'].index(ch)
                    channels_dict[ch].append(M[:, channel_index])

        Mean, SD, SE = {}, {}, {}
        for ch in self.channels:
            channel_data = np.array(channels_dict[ch])
            Mean[ch] = np.mean(channel_data, axis=0)
            SD[ch] = np.std(channel_data, axis=0)
            SE[ch] = SD[ch] / np.sqrt(N)

        return Mean, SD, SE, channels_dict
