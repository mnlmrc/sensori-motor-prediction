import json
import os
import globals as gl

import sys
from force import merge_blocks_mov, detect_state_change, force_segment
from fetch import save_npy, load_participants, load_mov

if __name__ == "__main__":
    # WAIT_TRIAL, // 0
    # START_TRIAL, // 1
    # WAIT_TR, // 2
    # WAIT_PLAN, // 3
    # WAIT_EXEC, // 4
    # GIVE_FEEDBACK, // 5
    # WAIT_ITI, // 6
    # ACQUIRE_HRF, // 7
    # END_TRIAL, // 8

    experiment = sys.argv[1]
    # folder = sys.argv[2]
    participant_id = sys.argv[2]

    filename = os.path.join(gl.baseDir, experiment, gl.behavDir, 'subj100', 'smp1_100_03.mov')

    mov = load_mov(filename)
    cols = ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
            'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz']


    # blocks = sys.argv[4]
    # blocks = blocks.split(",")
    # prestim = float(sys.argv[5])
    # poststim = float(sys.argv[6])
    # fsample = float(sys.argv[7])
    #
    # rawForce, states = merge_blocks_mov(experiment, folder, participant_id, blocks)
    # idx = detect_state_change(states)
    # force = force_segment(rawForce, idx, prestim=prestim, poststim=poststim, fsample=fsample)
    #
    # descr = json.dumps({
    #     'experiment': experiment,
    #     'folder': folder,
    #     'participant': participant_id,
    #     'fsample': fsample,
    #     'prestim': prestim,
    #     'poststim': poststim
    # })
    #
    # print(f"Saving participant {participant_id}...")
    # save_npy(force, descr, experiment, folder, participant_id)
    # print('Force saved!!!')
