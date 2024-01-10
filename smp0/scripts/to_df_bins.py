import sys

import numpy as np
import pandas as pd
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

    Infop = Info(
        experiment,
        participants=['100', '101', '102', '103', '104',
                      '105', '106', '107', '108', '110'],
        datatype=datatype,
        condition_headers=['stimFinger', 'cues']
    )

    Infof = Info(
        experiment,
        participants=['100', '101', '102', '103', '104',
                      '105', '106', '107', '108', '110'],
        datatype=datatype,
        condition_headers=['stimFinger']
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

        info_cols = ['ParticipantID', 'stimFinger', 'Win', 'Condition']
        channels = ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
            "pinkie_flex", "thumb_ext", "index_ext",
            "middle_ext", "ring_ext", "pinkie_ext", "fdi"]

        df = pd.DataFrame(columns=info_cols + channels)

        # create list of 3D data (binned force/EMG traces)
        for p, participant_id in enumerate(Infop.participants):
            data = load_npy(Infop.experiment, participant_id=participant_id, datatype=datatype)
            cond_vec_f = Infof.cond_vec[p]
            cond_vec_p = Infop.cond_vec[p]
            Zf = indicator(cond_vec_f).astype(bool)
            for f in range(Zf.shape[1]):
                bins = bin_traces(data[Zf[:, f]], wins, fsample=Param.fsample,
                                  offset=Param.prestim + np.mean(clamped_latency[f])).reshape(-1, len(Infop.channels[p]))
                cond_vec = cond_vec_p[Zf[:, f]].astype(int)
                # for w in range(bins.shape[-1]):
                #     bins_df = pd.DataFrame()
                #     for c, ch in enumerate(Infop.channels[p]):
                #         bins_df[ch] = pd.DataFrame(bins[:, c, w])
                #     bins_df['ParticipantID'] = participant_id
                #     bins_df['stimFinger'] = f
                #     bins_df['Win'] = f"{wins[w][0]}-{wins[w][1]}"
                #     bins_df['Condition'] = cond_vec
                #
                #     df[bins_df.columns] = pd.concat([df[bins_df.columns], bins_df], ignore_index=True)

                    # for col in bins_df.columns:
                    #     if col in df.columns:
                    #         df[col].
                    #
                    # df = pd.concat([df, bins_df], ignore_index=True)






        # plot_stim_aligned(M, SE, clamped_mean, clamped_latency, channels=channels[datatype], datatype=datatype)
    else:
        pass
