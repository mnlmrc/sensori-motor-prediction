import os
import warnings

import globals as gl
import numpy as np

from fetch import load_mov


def merge_blocks_mov(experiment=None, folder=None, participant_id=None, blocks=None):
    """

    :param experiment:
    :param participant_id:
    :param blocks: blocks list from field blocksForce in participants.tsv
    :return:
    """

    rawForce, states = [], []
    for block in blocks:

        print(f"loading participant: {participant_id} - block: {block}")

        rawF, st = load_mov(experiment, folder, participant_id, block)
        num_of_trials = len(st)

        for ntrial in range(num_of_trials):
            rawForce.append(rawF[ntrial])
            # vizF.append(vizForce[ntrial])
            states.append(st[ntrial])

    return rawForce, states


def detect_state_change(states, planState=3):
    """

    Args:
        states:
        planState:

    Returns:

    """
    idx = np.zeros(len(states)).astype(int)
    for st, state in enumerate(states):
        try:
            idx[st] = np.where(state > planState)[0][0]
        except:
            idx[st] = -1

    return idx


def force_segment(rawForce, idx, prestim=None, poststim=None, fsample=None):
    """

    :param rawForce:
    :param idx:
    :param prestim:
    :param poststim:
    :param fsample:
    :return:
    """

    ntrials = len(rawForce)
    nfingers = rawForce[0].shape[-1]
    timepoints = int(fsample * (prestim + poststim))

    force_segmented = np.zeros((ntrials, nfingers, timepoints))
    # NoResp = []
    for r, rawF in enumerate(rawForce):
        if idx[r] > 0:
            force_segmented[r] = (rawF[idx[r] - int(fsample * prestim):
                                       idx[r] + int(fsample * poststim)]).T
        else:
            pass

    return force_segmented
