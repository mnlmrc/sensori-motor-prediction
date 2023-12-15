import os
import warnings
import smp0.globals as gl

import numpy as np


def load_mov(experiment=None, participant_id=None, block=None):
    """
    load .mov file of one block

    :return:
    """
    fname = f"{experiment}_{participant_id}_{"{:02d}".format(int(block))}.mov"
    filepath = os.path.join(gl.make_dirs(experiment, participant_id, "mov"), fname)

    try:
        with open(filepath, 'rt') as fid:
            trial = 0
            A = []
            for line in fid:
                if line.startswith('Trial'):
                    trial_number = int(line.split(' ')[1])
                    trial += 1
                    if trial_number != trial:
                        warnings.warn('Trials out of sequence')
                        trial = trial_number
                    A.append([])
                else:
                    # Convert line to a numpy array of floats and append to the last trial's list
                    data = np.fromstring(line, sep=' ')
                    if A:
                        A[-1].append(data)
                    else:
                        # This handles the case where a data line appears before any 'Trial' line
                        warnings.warn('Data without trial heading detected')
                        A.append([data])

            # Convert all sublists to numpy arrays
            rawForce = [np.array(trial_data)[:, 4:9] for trial_data in A]
            # vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
            state = [np.array(trial_data)[:, 1] for trial_data in A]

    except IOError as e:
        raise IOError(f"Could not open {filepath}") from e

    return rawForce, state


def merge_blocks_mov(experiment=None, participant_id=None, blocks=None):
    """

    :param experiment:
    :param participant_id:
    :param blocks: blocks list from field blocksForce in participants.tsv
    :return:
    """

    rawForce, states = [], []
    for block in blocks:

        print(f"loading participant: {participant_id} - block: {block}")

        rawF, st = load_mov(experiment, participant_id, block)
        num_of_trials = len(st)

        for ntrial in range(num_of_trials):
            rawForce.append(rawF[ntrial])
            # vizF.append(vizForce[ntrial])
            states.append(st[ntrial])

    return rawForce, states


def detect_state_change(states):
    """

    :param states:
    :return:
    """
    idx = np.zeros(len(states)).astype(int)
    for st, state in enumerate(states):
        try:
            idx[st] = np.where(state > 2)[0][0]
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
