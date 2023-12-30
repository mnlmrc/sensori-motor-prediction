import sys

import numpy as np
from PcmPy import indicator

from smp0.experiment import load_dat, participants
from smp0.utils import load_npy, pool_participants
from smp0.visual import plot_response_by_probability


# def plot_single_participant():
#     Zi = Z_probability(d, stimFinger="index")
#     Zr = Z_probability(d, stimFinger="ring")
#     _, sorted_mean_i, condition = sort_by_probability(data_single, Zi)
#     _, sorted_mean_r, _ = sort_by_probability(data_single, Zr)
#     data_c = np.stack([sorted_mean_i, sorted_mean_r], axis=0)
#     plot_response_by_probability(data_c, clamped_mean, datatype=datatype)

if __name__ == "__main__":
    experiment = sys.argv[1]
    datatype = sys.argv[2]

    conditions_keys = ["stimFinger", "cues"]  # Update as needed

    if datatype == 'emg':
        channels_names = "muscle_names"
    elif datatype == 'mov':
        channels_names = "finger_names"
    else:
        channels_names = None
    # clamped = load_npy(experiment, 'clamped', 'mov')
    # clamped_d = load_dat(experiment, 'clamped')
    # clamped_Z = Z_probability(clamped_d.chordID, clamped_d.stimFinger)
    # clamped_i = clamped[clamped_Z[0, :, 0]].mean(axis=0)
    # clamped_r = clamped[clamped_Z[1, :, 1]].mean(axis=0)
    # clamped_mean = np.stack([clamped_i, clamped_r], axis=0)

    if len(sys.argv) == 4:
        participant_id = sys.argv[3]
        d = load_dat(experiment,
                     participant_id)
        data_single = load_npy(experiment=experiment,
                               participant_id=participant_id,
                               datatype=datatype)
        # plot_single_participant()
    elif len(sys.argv) == 3:
        data_pooled = pool_participants(experiment,
                                        conditions_keys,
                                        channels_names,
                                        datatype)
        plot_response_by_probability(data_pooled, datatype=datatype)

    else:
        pass
