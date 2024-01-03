import numpy as np

from smp0.load_and_save import load_participants, load_dat


class Experiment:
    def __init__(self, experiment):
        self.experiment = experiment
        self.info = load_participants(experiment)
        self.p_dict = {
            'mov': self.get_info('mov'),
            'emg': self.get_info('emg')
        }

    def get_info(self, datatype):
        """
        Transforms participants.tsv and .dat into a nested dictionary for efficient lookup.
        """
        p_dict = {}
        for index, row in self.info.iterrows():
            participant_id = row['participant_id'][-3:]
            if participant_id in participants:
                d = load_dat(self.experiment, participant_id)
                blocks = row[f"blocks_{datatype}"].split(",")
                blocks = [int(block) for block in blocks]
                d = d[d.BN.isin(blocks)]
                if participant_id not in p_dict:
                    p_dict[participant_id] = {}
                for column in self.info.columns:
                    if isinstance(row[column], str) and "," in row[column]:
                        p_dict[participant_id][column] = row[column].split(",")
                    else:
                        p_dict[participant_id][column] = row[column]
                p_dict[participant_id]['stimFinger'] = d.stimFinger.to_numpy()
                p_dict[participant_id]['cues'] = d.chordID.to_numpy()

        return p_dict


channels = {
    'mov': ["thumb", "index", "middle", "ring", "pinkie"],
    'emg': ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
            "pinkie_flex", "thumb_ext", "index_ext",
            "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
}

conditions = {
    'stimFinger': {
        "index": 91999,
        "ring": 99919
    },
    'cues': {
        "25%": 12,  # index 25% - ring 75% - 0
        "75%": 21,  # index 75% - ring 25% - 1
        "100%": 39,  # index 100% - ring 0% - 2
        "50%": 44,  # index 50% - ring 50% - 3
        "0%": 93  # index 0% - ring 100% - 4
    }
}





participants = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '110']

fsample = {'emg': 2148.1481,
           'mov': 500}
prestim = 1  # time before stimulation (s)
poststim = 2  # time after stimulation (s)
ampThreshold = 2
filter_nord = 4  # n_ord
filter_cutoff = 30  # cutoff frequency

timeS = {
    "emg": np.linspace(-prestim,
                       poststim,
                       int((prestim + poststim) * fsample['emg'])),
    "mov": np.linspace(-prestim,
                       poststim,
                       int((prestim + poststim) * fsample['mov']))
        }