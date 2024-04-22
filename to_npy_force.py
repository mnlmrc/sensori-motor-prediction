import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import globals as gl

import sys
from experiment import Param
from fetch import load_mov

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
    participant_id = sys.argv[2]
    session = sys.argv[3]

    # extract subject number
    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    # read participants.tsv
    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    # load segmentation params
    param = Param()
    fsample = param.fsample
    prestim = int(param.prestim * fsample)
    poststim = int(param.poststim * fsample)

    if session == 'scanning':
        blocks = participants[participants.sn == sn].runsSess1.iloc[0].split('.')
        path = os.path.join(gl.baseDir, experiment, gl.behavDir, participant_id)
    elif session == 'training':
        blocks = participants[participants.sn == sn].runsTraining.iloc[0].split('.')
        path = os.path.join(gl.baseDir, experiment, gl.trainDir, participant_id)
    else:
        raise ValueError('Session name not recognized. Allowed session names are "scanning" and "training".')

    dat = pd.read_csv(os.path.join(path, f'{experiment}_{sn}.dat'), sep='\t')

    force = list()
    trial_info = {
        'cue': list(),
        'stimFinger': list(),
        'trialLabel': list()
    }
    columns = ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
               'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz']

    for bl in blocks:
        print(f'processing... {participant_id}, block {bl}')
        block = '%02d' % int(bl)
        filename = os.path.join(path, f'{experiment}_{sn}_{block}.mov')

        mov = load_mov(filename)
        movC = np.concatenate(mov, axis=0)
        mov_df = pd.DataFrame(movC, columns=columns)

        bl_col = np.concatenate(
            [np.zeros(tr.shape[0]) + int(bl) for tr in mov])  # make a column that specify block number
        mov_df = pd.concat([pd.DataFrame(bl_col, columns=['block']), mov_df], axis='columns')

        planState = 3
        idx = (mov_df['state'] > planState).to_numpy().astype(int)
        idxD = np.diff(idx)
        stimOnset = np.where(idxD == 1)[0]

        for st, ons in enumerate(stimOnset):
            BN = mov_df['block'].iloc[ons]
            TN = mov_df['trialNum'].iloc[ons]
            GoNogo = dat[(dat.BN == BN) & (dat.TN == TN)].GoNogo.iloc[0]
            if GoNogo == 'go':
                force.append(movC[ons - prestim:ons + poststim])
                trial_info['cue'].append(dat[(dat.BN == BN) & (dat.TN == TN)]['cue'].iloc[0].astype(str))
                trial_info['stimFinger'].append(dat[(dat.BN == BN) & (dat.TN == TN)]['stimFinger'].iloc[0].astype(str))
                trial_info['trialLabel'].append(dat[(dat.BN == BN) & (dat.TN == TN)]['trialLabel'].iloc[0])

    descr = json.dumps({
        'experiment': experiment,
        'participant': participant_id,
        'fsample': fsample,
        'prestim': prestim,
        'poststim': poststim,
        'columns': columns,
        'trial_info': trial_info
    })

    print(f"Saving participant {participant_id}, session {session}...")
    np.savez(os.path.join(path, f'{experiment}_{sn}.npz'),
             data_array=force, descriptor=descr, allow_pickle=False)

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

    # print('Force saved!!!')
