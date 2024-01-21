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
    stimFinger = ['index', 'ring']

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

    Plot = Plotter3D(
        xAx=(xAx, xAx),
        data=Y,
        channels=channels[datatype],
        conditions=stimFinger,
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

    # statistics
    Anova = Anova3D(
        data=Y,
        channels=channels[datatype],
        conditions=stimFinger,
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
    rm_anova_cue_tp = rm_anova(df, ['channel', 'stimFinger'], ['cue', 'timepoint'])

    xTick = [str(group).strip("()").replace("'", "") + ", " + factor for group, factor in zip(rm_anova_cue_tp.group, rm_anova_cue_tp.factor)]
    fig, axs = plt.subplots(len(channels[datatype]), len(stimFinger),
                            figsize=(6.4, 8), sharey=True, sharex=True)
    significant = .05
    for xt, pval in zip(xTick, rm_anova_cue_tp.pval):
        ch = xt.split(", ")[0]
        sf = xt.split(", ")[1]
        fc = xt.split(", ")[-1]
        row = channels[datatype].index(ch)
        col = stimFinger.index(sf)
        axs[row, col].bar(fc, pval)
        axs[row, col].axhline(significant, ls='--', color='r')
        axs[row, col].set_title(ch)
    axs[0, 0].set_ylim([0, .1])
    fig.supylabel('p-pvalue')
    fig.tight_layout()

    rm_anova_cue = rm_anova(df, ['channel', 'stimFinger', 'timepoint'], ['cue'])
    xTick = [str(group).strip("()").replace("'", "") + ", " + factor for group, factor in
             zip(rm_anova_cue.group, rm_anova_cue.factor)]
    fig, axs = plt.subplots(len(channels[datatype]), len(stimFinger),
                            figsize=(6.4, 8), sharey=True, sharex=True)
    significant = .05
    for xt, pval in zip(xTick, rm_anova_cue.pval):
        ch = xt.split(", ")[0]
        sf = xt.split(", ")[1]
        tp = xt.split(", ")[2]
        fc = xt.split(", ")[-1]
        row = channels[datatype].index(ch)
        col = stimFinger.index(sf)
        axs[row, col].bar(tp, pval)
        axs[row, col].axhline(significant, ls='--', color='r')
        axs[row, col].set_title(ch)
    axs[0, 0].set_ylim([0, .1])
    fig.supylabel('p-pvalue')
    fig.tight_layout()
    plt.show()




