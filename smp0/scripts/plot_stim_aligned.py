import sys

import numpy as np
from PcmPy import indicator
from matplotlib import pyplot as plt

from smp0.experiment import Info, Clamped, Param
from smp0.fetch import load_npy
from smp0.utils import bin_traces
from smp0.visual import Plotter3D
from smp0.workflow import list_participants, av_across_participants

if __name__ == "__main__":
    experiment = sys.argv[1]
    datatype = sys.argv[2]
    plottype = sys.argv[3]

    Clamp = Clamped(experiment)

    Params = Param(datatype)

    Info_p = Info(
        experiment,
        participants=['100', '101', '102', '103', '104',
                      '105', '106', '107', '108', '110'],
        datatype=datatype,
        condition_headers=['stimFinger', 'cues']
    )

    Info_f = Info(
        experiment,
        participants=['100', '101', '102', '103', '104',
                      '105', '106', '107', '108', '110'],
        datatype=datatype,
        condition_headers=['stimFinger']
    )

    wins = ((-1, 0),
            (0, .05),
            (.05, .1),
            (.1, .5))

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

    if plottype == 'con':

        # create list of 3D data (segmented trials)
        Data = list()
        for participant_id in Info_p.participants:
            data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
            Data.append(data)

        # create list of participants
        Y = list_participants(Data, Info_p)

        timeAx = Params.timeAx()
        timeAx_c = (timeAx - Clamp.latency[0], timeAx - Clamp.latency[1])

        Plot = Plotter3D(
            xAx=timeAx_c,
            data=Y,
            channels=channels[datatype],
            conditions=['index', 'ring'],
            labels=['0%', '25%', '50%', '75%', '100%'],
            xlabel='time (s)',
            ylabel=ylabel[datatype],
            figsize=(6.4, 8),
            xlim=(-.1, .5)

        )

        colors = Plot.make_colors()
        Plot.subplots((colors[1:], colors[:4]))
        Plot.set_titles()
        Plot.legend(colors)
        Plot.xylabels()
        Plot.set_xylim()
        Plot.fig.set_constrained_layout(True)
        plt.show()

    elif plottype == 'bin':

        Data = list()
        for p, participant_id in enumerate(Info_p.participants):
            data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
            Zf = indicator(Info_f.cond_vec[p]).astype(bool)
            bins_i = bin_traces(data[Zf[:, 0]], wins, fsample=Params.fsample,
                                offset=Params.prestim + Clamp.latency[0])
            bins_r = bin_traces(data[Zf[:, 1]], wins, fsample=Params.fsample,
                                offset=Params.prestim + Clamp.latency[1])
            bins = np.concatenate((bins_i, bins_r), axis=0)
            Info_p.cond_vec[p] = np.concatenate((Info_p.cond_vec[p][Zf[:, 0]], Info_p.cond_vec[p][Zf[:, 1]]),
                                               axis=0).astype(int)
            Data.append(bins)

        # create list of participants
        Y = list_participants(Data, Info_p)

        # xAx = [f"{win[0]}s to {win[1]}s" for win in wins]
        xAx = np.linspace(0, 3, 4)

        # Plot = Plotter3D(
        #     xAx=(xAx, xAx),
        #     data=Y,
        #     channels=channels[datatype],
        #     conditions=['index', 'ring'],
        #     labels=['0%', '25%', '50%', '75%', '100%'],
        #     yla