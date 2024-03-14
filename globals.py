import os
from pathlib import Path


def make_dirs(experiment=None, folder=None, participant_id=None):
    """

    :param experiment: experiment code (e.g., smp0, smp1, ...)
    :param folder: folder that contains data (e.g., behavioural, training, emg, ...)
    :param participant_id: participant identifier
    :return:
    """

    directory = os.path.join(base_dir, experiment, folder, f"subj{participant_id}")

    # if datatype is not None and participant_id.isdigit():
    #     directory = os.path.join(base_dir, experiment, folder, f"subj{participant_id}")
    # elif datatype is not None and not participant_id.isdigit():
    #     directory = os.path.join(base_dir, experiment, folder, f"{participant_id}")
    # elif datatype is None and participant_id.isdigit():
    #     directory = os.path.join(base_dir, experiment, folder, f"subj{participant_id}")
    # elif datatype is None and not participant_id.isdigit():
    #     directory = os.path.join(base_dir, experiment, folder, f"{participant_id}")
    # else:
    #     directory = None

    return directory


# base_dir = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction'
# if not Path(base_dir).exists():
#     base_dir = '/content/drive/My Drive/UWO/SensoriMotorPrediction/'
#     print(base_dir)
# else:
#     print(base_dir)


base_dir = "/Volumes/MotorControl/data/SensoriMotorPrediction/"
if not Path(base_dir).exists():
    print("Switch to local directory")
    base_dir = ('/Users/mnlmrc/Library/CloudStorage/'
                'GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction')
    # base_dir = '/content/drive/My Drive/UWO/SensoriMotorPrediction'
print("Base directory:", base_dir)

