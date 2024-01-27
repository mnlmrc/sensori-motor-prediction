import sys

import numpy as np
from PcmPy import indicator
from matplotlib import pyplot as plt

from smp0.experiment import Info, Clamped, Param
from smp0.fetch import load_npy
# from smp0.stat import
from smp0.utils import bin_traces
from smp0.visual import Plotter, dict_vlines, dict_bars, dict_text, dict_lims, add_entry_to_legend, dict_legend
from smp0.workflow import list_participants3D, list_participants2D

if __name__ == "__main__":
    experiment = sys.argv[1]
    datatype = sys.argv[2]

    participants = ['100', '101', '102', '103', '104',
                    '105', '106', '107', '108', '110']

    Clamp = Clamped(experiment)
    Params = Param(datatype)
    Info_p = Info(experiment, participants, datatype, ['stimFinger', 'cues'])
    c_vec_f = Info(experiment, participants, datatype, ['stimFinger']).cond_vec

    wins = {
        'mov': ((-1, 0),
                (0, .1),
                (.1, .3),
                (.3, 1)),
        'emg': ((-1, 0),
                (0, .05),
                (.05, .1),
                (.1, .3))
    }
    wins = wins[datatype]
    win_syn = [(.05, .1)]

    # define channels to plot for each datatype
    channels = {
        'mov': ["thumb", "index", "middle", "ring", "pinkie"],
        'emg': ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
                "pinkie_flex", "thumb_ext", "index_ext",
                "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
    }

    # define ylabel per datatype
    ylabel = {
        'mov': 'force (N)',
        'emg': 'emg (mV)'
    }

    labels = ['0%', '25%', '50%', '75%', '100%']

    # create list of 3D data (segmented trials)
    Data = list()
    for participant_id in Info_p.participants:
        data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
        if datatype == 'emg':
            bins = bin_traces(data, (wins[0], ), fsample=Params.fsample,
                              offset=Params.prestim + Clamp.latency[0])
            data = data / bins
        Data.append(data)

    # create list of participants
    Y = list_participants3D(Data, Info_p)

    timeAx = Params.timeAx()
    timeAx_c = (timeAx - Clamp.latency[0], timeAx - Clamp.latency[1])

    dict_lims['xlim'] = (-.1, .5)
    dict_text['ylabel'] = ylabel[datatype]
    dict_legend['ncol'] = 6
    dict_vlines['pos'] = [win[1] for win in wins]
    dict_vlines['lw'] = [1] * len(wins)
    dict_vlines['color'] = ['k'] * len(wins)
    dict_vlines['ls'] = ['-', '-.', ':', '--']

    Plot = Plotter(
        xAx=timeAx_c,
        data=Y,
        channels=channels[datatype],
        conditions=['index', 'ring'],
        labels=labels,
        figsize=(6.4, 8),
        lims=dict_lims,
        legend=dict_legend,
        vlines=dict_vlines,
        text=dict_text
    )

    colors = Plot.make_colors()
    Mean, _, SE, _ = Plot.av_across_participants()
    Plot.subplots3D(Mean, SE, (colors[1:], colors[:4]))
    Plot.set_titles()
    Plot.set_legend(colors)
    Plot.xylabels()
    Plot.set_xyticklabels_size()
    Plot.set_xylim()
    Plot.add_vertical_lines()
    # Plot.fig.set_constrained_layout(True)
    Plot.fig.subplots_adjust(hspace=.5, bottom=.06, top=.95, left=.1, right=.9)

    if datatype == 'mov':
        for row in range(Plot.axs.shape[0]):
            for col in range(Plot.axs.shape[1]):
                Plot.axs[row, col].plot(Clamp.timeAx()[col], Clamp.clamped_f[col],
                                        ls='--', color='k', lw=1)
        Plot.axs[0, 0].plot(np.nan, ls='--', color='k', lw=1)
        add_entry_to_legend(Plot.fig, label='clamped')

    plt.show()
