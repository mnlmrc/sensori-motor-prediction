import sys

import numpy as np
from PcmPy import indicator

from dataset import Dataset3D
from smp0.experiment import Info
from smp0.fetch import load_dat, load_npy

from smp0.utils import detect_response_latency
from smp0.visual import plot_stim_aligned
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

    if len(sys.argv) == 4:
        pass
        # plot_single_participant()
    elif len(sys.argv) == 3:

        # create list of 3D data (segmented trials)
        Data = list()
        for participant_id in Info.participants:
            data = load_npy(Info.experiment, participant_id=participant_id, datatype=datatype)
            Data.append(data)

        # create list of participants
        Y = create_participants_list3D(Data, Info)

        # define channels to plot for each datatype
        channels = {
            'mov': ["thumb", "index", "middle", "ring", "pinkie"],
            'emg': ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
                    "pinkie_flex", "thumb_ext", "index_ext",
                    "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
        }

        # calculate descriptives across participants for each channel
        M, SD, SE, _ = create_channels_dictionary(Y, channels=channels[datatype])

        # plot channels
        plot_stim_aligned(M, SE, clamped_mean, clamped_latency, channels=channels[datatype], datatype=datatype)
    else:
        pass
