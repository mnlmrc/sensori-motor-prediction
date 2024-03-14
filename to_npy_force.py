import json

import sys
from force import merge_blocks_mov, detect_state_change, force_segment
from fetch import save_npy, load_participants

if __name__ == "__main__":
    experiment = sys.argv[1]
    folder = sys.argv[2]
    participant_id = sys.argv[3]
    blocks = sys.argv[4]
    blocks = blocks.split(",")
    prestim = float(sys.argv[5])
    poststim = float(sys.argv[6])
    fsample = float(sys.argv[7])

    rawForce, states = merge_blocks_mov(experiment, folder, participant_id, blocks)
    idx = detect_state_change(states)
    force = force_segment(rawForce, idx, prestim=prestim, poststim=poststim, fsample=fsample)

    descr = json.dumps({
        'experiment': experiment,
        'folder': folder,
        'participant': participant_id,
        'fsample': fsample,
        'prestim': prestim,
        'poststim': poststim
    })

    print(f"Saving participant {participant_id}...")
    save_npy(force, descr, experiment, folder, participant_id)
    print('Force saved!!!')
