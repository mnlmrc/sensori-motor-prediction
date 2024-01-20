import sys

import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from PcmPy import indicator
from matplotlib import pyplot as plt

from smp0.experiment import Info, Clamped, Param
from smp0.fetch import load_npy
from smp0.stat import Anova3D, rm_anova
from smp0.utils import bin_traces, nnmf, assign_synergy, split_column_df
from smp0.visual import Plotter3D, dict_vlines, dict_bars, dict_text, dict_lims, add_entry_to_legend, dict_legend
from smp0.workflow import list_participants3D, list_participants2D

if __name__ == "__main__":
    experiment = sys.argv[1]
    datatype = sys.argv[2]

    participants = ['100', '101', '102', '103', '104',
                      '105', '106', '107', '108', '109','110']

    Clamp = Clamped(experiment)
    Params = Param(datatype)
    Info_p = Info(experiment, participants, datatype, ['stimFinger', 'cues'])
    c_vec_f = Info(experiment, participants, datatype,  ['stimFinger']).cond_vec

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

    Data = list()
    for p, participant_id in enumerate(Info_p.participants):
        data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
        Zf = indicator(c_vec_f[p]).astype(bool)
        bins_i = bin_traces(data[Zf[:, 0]], wins, fsample=Params.fsample,
                            offset=Params.prestim + Clamp.latency[0])
        bins_r = bin_traces(data[Zf[:, 1]], wins, fsample=Params.fsample,
                            offset=Params.prestim + Clamp.latency[1])
        bins = np.concatenate((bins_i, bins_r), axis=0)
        Info_p.cond_vec[p] = np.concatenate((Info_p.cond_vec[p][Zf[:, 0]], Info_p.cond_vec[p][Zf[:, 1]]),
                                            axis=0).astype(int)
        Data.append(bins)

    # create list of participants
    Y = list_participants3D(Data, Info_p)

    xAx = np.linspace(0, 3, 4)

    dict_lims['xlim'] = (-1, 4)
    dict_text['xlabel'] = None
    dict_text['ylabel'] = ylabel[datatype]
    dict_text['xticklabels'] = [f"{win[0]}s to {win[1]}s" for win in wins]

    Anova = Anova3D(
        data=Y,
        channels=channels[datatype],
        conditions=['index', 'ring'],
        labels=labels
    )

    df = Anova.make_df(labels=[
        'index, 25%',
        'index, 50%',
        'index, 75%',
        'index, 100%',
        'ring, 0%',
        'ring, 25%',
        'ring, 50%',
        'ring, 100%',
    ])
    df = split_column_df(df, ['stimFinger', 'cue'], 'condition')
    results_df = rm_anova(df, ['channel', 'stimFinger',], ['cue', 'timepoint'])
    print(results_df)

    # _, _, _, ch_dict = Anova.av_across_participants()
    # rm_anova_dict = {ch: None for ch in channels[datatype]}
    # for ch in channels[datatype]:
    #     _, _, _, pval = Anova.rm_anova(ch_dict[ch], (labels[1:], labels[:4]))
    #     rm_anova_dict[ch] = pval

    Plot = Plotter3D(
        xAx=(xAx, xAx),
        data=Y,
        channels=channels[datatype],
        conditions=['index', 'ring'],
        labels=['0%', '25%', '50%', '75%', '100%'],
        lims=dict_lims,
        text=dict_text,
        bar=dict_bars,
        figsize=(6.4, 8),
        plotstyle='bar'
    )

    colors = Plot.make_colors()
    Plot.subplots((colors[1:], colors[:4]))
    Plot.set_titles()
    Plot.set_legend(colors)
    Plot.xylabels()
    Plot.set_xylim()
    Plot.set_xyticklabels_size()
    Plot.set_xticklabels()
    # Plot.fig.set_constrained_layout(True)
    Plot.fig.subplots_adjust(hspace=.5, bottom=.08, top=.95, left=.1, right=.9)

    # add pvals to bars
    # ytext = Plot.axs[0, 0].get_ylim()[1]
    # for row, channel in enumerate(channels[datatype]):
    #     for col in range(Plot.axs.shape[1]):
    #         for xtext in xAx:
    #             pval = rm_anova_dict[channel][col, int(xtext)]
    #             Plot.axs[row, col].text(xtext, ytext, f"p={pval:.2f}",
    #                                     ha='center', va='center', fontsize=6)
    # Plot.axs[0, 0].set_ylim([None, ytext + .05 * ytext])

    plt.show()




