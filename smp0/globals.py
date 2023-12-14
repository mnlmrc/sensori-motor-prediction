import os
from pathlib import Path


def make_dirs(experiment=None, participant_id=None, datatype=None):
    """

    :param experiment:
    :param participant_id:
    :param datatype:
    :return:
    """

    if datatype is not None:
        _dir = os.path.join(base_dir, experiment, f"subj{participant_id}", datatype)
    else:
        _dir = os.path.join(base_dir, experiment, f"subj{participant_id}")

    return _dir


base_dir = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction'
if not Path(base_dir).exists():
    pass
if not Path(base_dir).exists():
    pass
