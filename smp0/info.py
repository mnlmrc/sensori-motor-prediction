import os

import pandas as pd

import smp0.globals as gl


def load_participants(experiment):
    """

    :param experiment:
    :return:
    """
    filepath = os.path.join(gl.base_dir, experiment, "participants.tsv")
    fid = open(filepath, 'rt')
    participants = pd.read_csv(fid, delimiter='\t', engine='python')

    return participants


def load_dat(experiment, participant_id):
    """

    :param experiment:
    :param participant_id:
    :return:
    """
    fname = f"{experiment}_{participant_id}.dat"
    filepath = os.path.join(gl.make_dirs(experiment, participant_id), fname)

    try:
        fid = open(filepath, 'rt')
        D = pd.read_csv(fid, delimiter='\t', engine='python')
    except IOError as e:
        raise IOError(f"Could not open {filepath}") from e

    return D


task = {
    "stim_finger": ["ring", "index"],
    "cues": ["0%", "25%", "50%", "75%", "100%"],

}
