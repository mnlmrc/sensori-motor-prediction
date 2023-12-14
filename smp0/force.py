import os
import warnings

import numpy as np


def load_mov(experiment=None, participant_id=None, block=None):
    """
    load .mov file of one block

    :param experiment:
    :param participant_id:
    :param block:
    :return:
    """
    fname = f"{experiment}_{participant_id}_{'{ :02d}'.format(block)}.mov"
    filepath = os.path.join(self.path, self.experiment, f"subj{self.participant_id}", 'mov', fname)

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
        raise IOError(f"Could not open {fname}") from e

    return rawForce, state


def merge_blocks_mov(experiment, participant_id, blocks):
    """

    :param experiment:
    :param participant_id:
    :param blocks: blocks list from field blocksForce in participants.tsv
    :return:
    """

    rawForce = []
    state = []
    for block in blocks:

        print(f"loading participant: {participant_id} - block: {block}")

        rawF, st = load_mov(experiment, participant_id, block)
        num_of_trials = len(st)

        for ntrial in range(num_of_trials):
            rawForce.append(rawF[ntrial])
            # vizF.append(vizForce[ntrial])
            state.append(st[ntrial])

    return rawForce, state


def detect_state_change(state):
    for ntrial in range(self.ntrials * len(self.blocks)):
        try:


def force_segment(rawF, state, ):

    force = np.zeros(
        (self.ntrials * len(self.blocks), num_chan, fsample * (self.prestim + self.poststim)))
    NoResp = []
    for ntrial in range(self.ntrials * len(self.blocks)):
        try:
            stim_idx = np.where(state[ntrial] > 2)[0][0]
            force[ntrial] = (rawF[ntrial][stim_idx - Force.fsample * self.prestim:
                                               stim_idx + Force.fsample * self.poststim]).T
        except:
            NoResp.append(ntrial + 1)

    return force, NoResp
