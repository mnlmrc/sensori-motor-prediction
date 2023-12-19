import sys

from smp0.force import merge_blocks_mov, detect_state_change, force_segment
from smp0.load_and_save import load_participants, save_npy
from smp0.utils import vlookup_value

if __name__ == "__main__":
    experiment = sys.argv[1]
    participant_id = sys.argv[2]

    prestim = 1
    poststim = 2
    fsample = 500

    blocks = None
    if len(sys.argv) == 3:
        info = load_participants(experiment)
        blocks = vlookup_value(info,
                               'participant_id',
                               f"subj{participant_id}",
                               'blocksForce').split(",")
    elif len(sys.argv) == 4:
        blocks = sys.argv[3].split(",")

    rawForce, states = merge_blocks_mov(experiment,
                                        participant_id,
                                        blocks)
    idx = detect_state_change(states)
    npy_force = force_segment(rawForce,
                              idx,
                              prestim=prestim,
                              poststim=poststim,
                              fsample=fsample)

    print(f"Saving participant {participant_id}...")
    save_npy(npy_force, experiment, participant_id, datatype='mov')
    print('EMG saved!!!')
