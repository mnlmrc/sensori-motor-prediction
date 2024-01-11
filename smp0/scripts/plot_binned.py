import sys

import numpy as np
from PcmPy import indicator

from smp0.experiment import Info, Param
from smp0.fetch import load_dat, load_npy

from smp0.utils import detect_response_latency, bin_traces
from smp0.visual import plot_stim_aligned, plot_binned
from smp0.workflow import create_participants_list3D, create_channels_dictionary, process_clamped

if __name__ == "__main__":
    experiment = sys.argv[1]
    datatype = sys.argv[2]

    clamped_mean, clamped_latency = process_clamped(experiment)

    Info = Info(
        experiment,
        participants=['100', '101', '102', '103', '104',
                      '105', '106', '107', '108', '110'],
        datatype=datatype,
        condition_headers=['stimFinger', 'cues']
    )

    Param = Param(datatype)

    wins = ((-1, 0),
            (0, .05),
            (.05, .1),
            (.1, .5))

    if len(sys.argv) == 4:
        pass
        # plot_single_participant()
    elif len(sys.argv) == 3:

        # create list of 3D data (binned force/EMG traces)
        Data = list()
        for participant_id in Info.participants:
            data = load_npy(Info.experiment, participant_id=participant_id, datatype=datatype)
            bins = bin_traces(data, wins, fsample=Param.fsample, offset=Param.prestim + np.mean(clamped_latency))
            Data.append(bins)

        # plot_stim_aligned(M, SE, clamped_mean, clamped_latency, channels=channels[datatype], datatype=datatype)
    else:
        pass
