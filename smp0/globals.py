import os
from pathlib import Path


def make_dirs(experiment=None, participant_id=None, datatype=None):
    """

    :param experiment:
    :param participant_id:
    :param datatype:
    :return:
    """

    if datatype is not None and participant_id.isdigit():
        directory = os.path.join(base_dir, experiment, f"subj{participant_id}", datatype)
    elif datatype is not None and not participant_id.isdigit():
        directory = os.path.join(base_dir, experiment, f"{participant_id}", datatype)
    elif datatype is None and participant_id.isdigit():
        directory = os.path.join(base_dir, experiment, f"subj{participant_id}")
    elif datatype is None and not participant_id.isdigit():
        directory = os.path.join(base_dir, experiment, f"{participant_id}")
    else:
        directory = None

    return directory


base_dir = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction'
if not Path(base_dir).exists():
    base_dir = '/content/drive/My Drive/UWO/SensoriMotorPrediction/'
    print(base_dir)
else:
    print(base_dir)

