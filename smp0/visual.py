import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from PcmPy import indicator

from smp0.experiment import Param
from smp0.workflow import av_within_participant

# from smp0.utils import detect_response_latency

# matplotlib.use('MacOSX')

dict_text = {
    'xlabel': 'time (s)',
    'ylabel': None,
    'fs_label': 9,
    'xticklabels': None,
    'xticklabels_rotation': 45,
    'xticklabels_alignment': 'right',
    'fs_ticklabels': 7,
    'fs_title': 7,
    'fs_suptitle': 9,
}

dict_legend = {
    'fs': 6,
    'loc': 'upper center',
    'ncol': 5
}

dict_vlines = {
    'pos': list(),
    'ls': list(),
    'lw': list(),
    'color': list()
}

dict_lims = {
    'xlim': (None, None),
    'ylim': (None, None)
}

dict_bars = {
    'width': .2,
    'offset': .2
}


class Plotter3D:

    def __init__(self, xAx, data, channels=None, conditions=None, labels=None,
                 vlines=None, text=None, lims=None, bar=None, legend=None,
                 extreme_colors=('red', 'blue'), figsize=(6.4, 4.8),
                 plotstyle='plot', ):

        if legend is None:
            self.legend = dict_legend
        else:
            self.legend = legend
        if bar is None:
            self.bar = dict_bars
        else:
            self.bar = bar

        if lims is None:
            self.lims = dict_lims
        else:
            self.lims = lims

        if text is None:
            self.text = dict_text
        else:
            self.text = text

        if vlines is None:
            self.vlines = dict_vlines
        else:
            self.vlines = vlines

        self.xAx = xAx
        self.data = data
        self.channels = channels
        self.conditions = conditions
        self.labels = labels

        self.ecol = extreme_colors
        self.figsize = figsize

        self.plotstyle = plotstyle

    def add_vertical_lines(self):
        pos = self.vlines['pos']
        ls = self.vlines['ls']
        cl = self.vlines['color']
        lw = self.vlines['lw']
        for v, vl in enumerate(pos):
            for row in range(self.axs.shape[0]):
                for col in range(self.axs.shape[1]):
                    self.axs[row, col].axvline(vl, ls=ls[v], color=cl[v], lw=lw[v])

    def set_xticklabels(self):
        n_conditions = len(self.conditions)
        xticklabels = self.text['xticklabels']
        rot = self.text['xticklabels_rotation']
        ha = self.text['xticklabels_alignment']
        for c in range(n_conditions):
            self.axs[-1, c].set_xticks(self.xAx[c])
            self.axs[-1, c].set_xticklabels(xticklabels, rotation=rot, ha=ha)

    def set_xyticklabels_size(self):
        n_conditions = len(self.conditions)
        n_channels = len(self.channels)
        for col in range(n_conditions):
            for row in range(n_channels):
                self.axs[row, col].tick_params(axis='x', labelsize=self.text['fs_ticklabels'])
                self.axs[row, col].tick_params(axis='y', labelsize=self.text['fs_ticklabels'])

    def set_xylim(self):
        self.axs[0, 0].set_xlim(self.lims['xlim'])
        self.axs[0, 0].set_ylim(self.lims['ylim'])

    def xylabels(self):
        self.fig.supxlabel(self.text['xlabel'], fontsize=self.text['fs_label'])
        self.fig.supylabel(self.text['ylabel'], fontsize=self.text['fs_label'])

    def set_legend(self, colors):
        for color, label in zip(colors, self.labels):
            if self.plotstyle == 'plot':
                self.axs[0, 0].plot(np.nan, label=label, color=color)
            elif self.plotstyle == 'bar':
                self.axs[0, 0].bar(np.nan, np.nan, label=label, color=color)
        self.fig.legend(ncol=self.legend['ncol'], fontsize=self.legend['fs'], loc=self.legend['loc'])

    def set_titles(self):
        n_conditions = len(self.conditions)
        for ch, channel in enumerate(self.channels):
            for c in range(n_conditions):
                self.axs[ch, c].set_title(channel, fontsize=self.text['fs_title'], pad=3)

    def subplots(self, colors):
        n_conditions = len(self.conditions)
        # n_channels = len(self.channels)
        self._setup_subplots()
        Mean, SD, SE, channels_dict = self._av_across_participants()
        for ch, channel in enumerate(self.channels):
            for c in range(n_conditions):
                for i in range(Mean[channel].shape[-2]):
                    if self.plotstyle == 'plot':
                        self.axs[ch, c].plot(self.xAx[c], Mean[channel][c, i], color=colors[c][i])
                        self.axs[ch, c].fill_between(self.xAx[c],
                                                     Mean[channel][c, i] - SE[channel][c, i],
                                                     Mean[channel][c, i] + SE[channel][c, i],
                                                     color=colors[c][i], alpha=.2, lw=0)
                    elif self.plotstyle == 'bar':
                        self.axs[ch, c].bar(
                            self.xAx[c] + (i - Mean[channel].shape[-2] / 2) * self.bar['offset'] +
                            self.bar['width'] / 2,
                            Mean[channel][c, i], color=colors[c][i], yerr=SE[channel][c, i],
                            width=self.bar['width'])
                    else:
                        pass

    def _setup_subplots(self):
        n_conditions = len(self.conditions)
        n_channels = len(self.channels)
        self.fig, self.axs = plt.subplots(n_channels, n_conditions,
                                          figsize=self.figsize, sharey=True, sharex=True)
        if n_channels == 1 or n_conditions == 1:
            self.axs = np.array([self.axs])

    def make_colors(self):

        n_labels = len(self.labels)
        cmap = mcolors.LinearSegmentedColormap.from_list(f"{self.ecol[0]}_to_{self.ecol[1]}",
                                                         [self.ecol[0], self.ecol[1]], N=100)
        norm = plt.Normalize(0, n_labels)
        colors = [cmap(norm(i)) for i in range(n_labels)]

        return colors

    def _av_across_participants(self):

        n_conditions = len(self.conditions)

        channels_dict = {ch: [] for ch in self.channels}
        N = len(self.data)
        for p_data in self.data:
            Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
            M = av_within_participant(p_data.measurements, Z)

            for ch in self.channels:
                if ch in p_data.channel_descriptors['channels']:
                    channel_index = p_data.channel_descriptors['channels'].index(ch)
                    channels_dict[ch].append(M[:, channel_index])

        Mean, SD, SE = {}, {}, {}
        for ch in self.channels:
            channel_data = np.array(channels_dict[ch])
            channels_dict[ch] = channel_data
            Mean[ch] = np.mean(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                              channel_data.shape[2]))
            SD[ch] = np.std(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                           channel_data.shape[2]))
            SE[ch] = (SD[ch] / np.sqrt(N)).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                    channel_data.shape[2]))

        return Mean, SD, SE, channels_dict


def add_entry_to_legend(fig, label, color='k', ls='--'):
    # Check if there is an existing legend
    if fig.legends:
        leg = fig.legends[0]  # Assuming there's only one legend
        existing_handles = leg.legendHandles
        existing_labels = [text.get_text() for text in leg.get_texts()]

        # Extract properties from the existing legend
        loc = leg._loc
        ncol = leg._ncols
        fontsize = leg.get_texts()[0].get_fontsize()

        # Remove the old legend
        leg.remove()
    else:
        existing_handles, existing_labels = [], []
        loc, ncol, fontsize = 'best', 1, 'medium'  # Default values

    # Add the new entry
    new_handle = mlines.Line2D([], [], color=color, linestyle=ls)
    existing_handles.append(new_handle)
    existing_labels.append(label)

    # Create a new legend with the same properties
    fig.legend(handles=existing_handles, labels=existing_labels, loc=loc, ncol=ncol, fontsize=fontsize)

