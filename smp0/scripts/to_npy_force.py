import sys

from smp0.experiment import Exp
from smp0.force import merge_blocks_mov, detect_state_change, force_segment
from smp0.load_and_save import save_npy

if __name__ == "__main__":
    experiment = sys.argv[1]
    participant_id = sys.argv[2]

    MyExp = Exp(experiment)

    prestim = MyExp.prestim
    poststim = MyExp.poststim
    fsample = MyExp.fsample_mov

    blocks = None
    if len(sys.argv) == 3:
        blocks = MyExp.get_info()[f"subj{participant_id}"]['blocks_mov']
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
