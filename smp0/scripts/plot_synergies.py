import sys

import numpy as np
from PcmPy import indicator
from matplotlib import pyplot as plt

from smp0.experiment import Info, Clamped, Param
from smp0.fetch import load_npy
from smp0.stat import Anova
from smp0.utils import bin_traces, nnmf, assign_synergy
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

    Data = list()
    Synergies = list()
    R_squared = np.zeros(len(Info_p.participants))
    for p, participant_id in enumerate(Info_p.participants):
        data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
        Zf = indicator(c_vec_f[p]).astype(bool)
        bins_i = bin_traces(data[Zf[:, 0]], win_syn, fsample=Params.fsample,
                            offset=Params.prestim + Clamp.latency[0])
        bins_r = bin_traces(data[Zf[:, 1]], win_syn, fsample=Params.fsample,
                            offset=Params.prestim + Clamp.latency[1])
        bins = np.concatenate((bins_i, bins_r), axis=0)
        Info_p.cond_vec[p] = np.concatenate((Info_p.cond_vec[p][Zf[:, 0]], Info_p.cond_vec[p][Zf[:, 1]]),
                                            axis=0).astype(int)

        H_pred = np.array([np.zeros(len(Info_p.channels[p])), np.zeros(len(Info_p.channels[p]))])
        H_pred[0, Info_p.channels[p].index('index_flex')] = 1
        H_pred[1, Info_p.channels[p].index('ring_flex')] = 1

        W, H, R_squared[p] = nnmf(bins.squeeze())
        W, H = assign_synergy(W, H, H_pred)

        Synergies.append(H)
        Data.append(W)

    print(f"mean $R^2$: {R_squared.mean()}")

    Info_p.set_manual_channels(['index', 'ring'])
    Y = list_participants2D(Data, Info_p)

    xAx = 0

    dict_lims['xlim'] = (-1, 1)
    dict_text['xlabel'] = None
    dict_text['ylabel'] = ylabel[datatype]
    dict_text['xticklabels'] = [f"{win[0]}s to {win[1]}s" for win in win_syn]

    Plot = Plotter3D(
        xAx=(xAx, xAx),
        data=Y,
        channels=['index', 'ring'],
        conditions=['index', 'ring'],
        labels=['0%', '25%', '50%', '75%', '100%'],
        lims=dict_lims,
        text=dict_text,
        bar=dict_bars,
        figsize=(6.4, 7),
        plotstyle='bar'
    )

    colors = Plot.make_colors()
    Plot.subplots((colors[1:], colors[:4]))
    Plot.set_titles()
    Plot.set_legend(colors)
    Plot.xylabels()
    Plot.set_xylim()
    # Plot.set_xyticklabels_size()
    # Plot.set_xticklabels()
    # Plot.fig.set_constrained_layout(True)
    # Plot.fig.subplots_adjust(hspace=.5, bottom=.08, top=.95, left=.1, right=.9)
    plt.show()


