import sys

import numpy as np

from smp0.emg import emg_hp_filter, emg_rectify, emg_segment, detect_trig
from smp0.load_and_save import load_delsys, load_participants,save_npy
from smp0.utils import vlookup_value

if __name__ == "__main__":
    experiment = sys.argv[1]
    participant_id = sys.argv[2]

    info = load_participants(experiment)
    muscle_names = vlookup_value(info, 'participant_id', f"subj{participant_id}", 'muscle_names').split(",")
    blocks = vlookup_value(info, 'participant_id', f"subj{participant_id}", 'blocks_emg').split(",")

    trigger_name = "trigger"
    ntrials = 20 * len(blocks)

    n_ord = 4
    cutoff = 30

    amp_threshold = 2
    prestim = 1
    poststim = 2

    fsample = 2148.1481  # sampling rate EMG

    npy_emg = None

    for block in blocks:

        print(f"processing participant {participant_id} - block {block}")

        df_emg = load_delsys(experiment,
                             participant_id,
                             block,
                             muscle_names=muscle_names,
                             trigger_name=trigger_name)
        df_emg_filtered = emg_hp_filter(df_emg,
                                        n_ord=n_ord,
                                        cutoff=cutoff,
                                        fsample=fsample,
                                        muscle_names=muscle_names)
        df_emg_rectified = emg_rectify(df_emg,
                                       muscle_names=muscle_names)
        _, timestamp = detect_trig(df_emg["trigger"],
                                   df_emg["time"],
                                   amp_threshold=amp_threshold,
                                   ntrials=20)
        npy_emg_segmented = emg_segment(df_emg_rectified,
                                        timestamp,
                                        prestim=prestim,
                                        poststim=poststim,
                                        fsample=fsample)

        npy_emg = npy_emg_segmented if npy_emg is None else np.concatenate((npy_emg, npy_emg_segmented), axis=0)

    print(f"Saving participant {participant_id}...")
    save_npy(npy_emg, experiment, participant_id, datatype='emg')
    print('EMG saved!!!')
