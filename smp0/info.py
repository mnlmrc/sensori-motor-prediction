import os

import numpy as np
import pandas as pd

from smp0.load_and_save import load_participants
from smp0.utils import vlookup_value

task = {
    "stimFinger": {
        "index": 91999,
        "ring": 99919
    },
    "cues": {
        "0%": 93,  # index 0% - ring 100% - 4
        "25%": 12,  # index 25% - ring 75% - 0
        "50%": 44,  # index 50% - ring 50% - 3
        "75%": 21,  # index 75% - ring 25% - 1
        "100%": 39  # index 100% - ring 0% - 2
    },
    "Fingers": ["thumb", "index", "middle", "ring", "pinkie"],
    "Muscles": ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
                "pinkie_flex", "thumb_ext", "index_ext",
                "middle_ext", "ring_ext", "pinkie_ext", "fdi"],

}

smp0_info = load_participants('smp0')
smp0_participants = [
    '100',
    '101',
    '102',
    '103',
    '104',
    '105',
    '106',
    '107',
    '108',
    '110',
]

smp0_dict = {participant_id: {"blocks_mov": vlookup_value(smp0_info,
                                                        'participant_id',
                                                        f"subj{participant_id}",
                                                        'blocks_mov').split(","),
                            "blocks_emg": vlookup_value(smp0_info,
                                                        'participant_id',
                                                        f"subj{participant_id}",
                                                        'blocks_emg').split(","),
                            "muscle_names": vlookup_value(smp0_info,
                                                          'participant_id',
                                                          f"subj{participant_id}",
                                                          'muscle_names').split(",")
                            }
           for participant_id in smp0_participants}

params = {
    "fsample_emg": 2148.1481,
    "fsample_force": 500,
    "pre/poststim": (1, 2),  # time before and after stimulation (s)
    "ampThreshold": 2,
    "filter": (4, 30)  # n_ord and cutoff frequency
}

timeS = {
    "emg": np.linspace(-params["pre/poststim"][0],
                       params["pre/poststim"][1],
                       int((params["pre/poststim"][0] +
                            params["pre/poststim"][1]) * params["fsample_emg"])),
    "mov": np.linspace(-params["pre/poststim"][0],
                       params["pre/poststim"][1],
                       int((params["pre/poststim"][0] +
                            params["pre/poststim"][1]) * params["fsample_force"]))
}
