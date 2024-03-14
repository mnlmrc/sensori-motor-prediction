import numpy as np
from PcmPy import indicator

from fetch import load_participants, load_dat, load_npy
from utils import remap_chordID, detect_response_latency


def remap_code_to_condition(cond_vec, d, cond_headers):
    cond_map = list()
    unis = np.unique(cond_vec).astype(int)
    for u in unis:
        mapping = f"{u}"
        indices = np.where(cond_vec == u)[0]
        for cond in cond_headers:
            ori = np.unique(d[cond][indices])[0]
            if ori.size > 1:
                raise ValueError('Inconsistent condition label')
            mapping = mapping + f",{ori}"

        cond_map.append(mapping)

    return cond_map


class Info:

    def __init__(self, experiment, participants, folder=None, condition_headers=None, demographics=None):
        """

        Args:
            experiment:
            participants:
            folder:
            condition_headers:
            demographics:
        """

        self.experiment = experiment
        self._info = load_participants(self.experiment)  # load info from participants.tsv
        self.participants = participants

        if folder is not None:
            self.folder = folder
            self.condition_headers = condition_headers
            self.cond_vec, self.channels, self.n_trials, self.cond_map = self._process_participant_info()

        if demographics is not None:
            self.dem_info = demographics
            self.demographics = self._process_participant_demographics()

    def _process_participant_demographics(self):
        """
        Processes data for a specific participant and datatype.

        Parameters:
        info (DataFrame): DataFrame containing participant information.
        participant_id (str): The ID of the participant.
        datatype (str): The datatype to process.

        Returns:
        dict: A dictionary containing processed data for the specified participant and datatype.
        """
        participant_dict = {col: [] for col in self.dem_info}
        for column in self.dem_info:
            for participant_id in self.participants:
                piece = self._info.at[f"subj{participant_id}", column]
                piece = int(piece) if piece.isdigit() else piece
                participant_dict[column].append(piece)

        return participant_dict

    def _process_participant_info(self):

        # allocate memory
        n_trials = np.zeros(len(self.participants))
        cond_vec = list()
        channels = list()
        cond_map = list()

        # loop through participants
        for p, participant_id in enumerate(self.participants):

            # load .dat
            d = remap_chordID(load_dat(self.experiment, participant_id))

            # filter .dat using blocks available per datatype
            blocks = self._info.at[f"subj{participant_id}", f"blocks_{self.folder}"].split(",")
            blocks = [int(block) for block in blocks]
            d = d[d.BN.isin(blocks)]

            # store number of trials
            n_trials[p] = len(d)

            # create condition vector
            c_vec = np.zeros(len(d))
            for cond in self.condition_headers:
                c_vec = c_vec + d[cond].to_numpy()  # add error if ambiguity
            cond_vec.append(c_vec)

            # store recorded channels
            channels.append(self._info.at[f"subj{participant_id}", f"channels_{self.folder}"].split(","))

        cond_map = remap_code_to_condition(cond_vec[0], remap_chordID(load_dat(self.experiment, self.participants[0])),
                                           self.condition_headers)

        return cond_vec, channels, n_trials, cond_map

    def set_manual_channels(self, channels=list()):

        for p in range(len(self.participants)):
            self.channels[p] = channels


class Param:

    def __init__(self, folder=None, fsample=500, prestim=1, poststim=2):
        self.fsample = fsample
        self.fsample = self._fsample[folder]
        self.folder = folder
        self.prestim = prestim
        self.poststim = poststim

    def timeAx(self):
        """
        Generates a time series for the specified datatype.

        Parameters:
        datatype (str): The datatype to generate the time series for.

        Returns:
        numpy.ndarray: A NumPy array representing the time series.
        """
        return np.linspace(-self.prestim, self.poststim,
                           int((self.prestim + self.poststim) * self.fsample))


class Clamped:

    def __init__(self, experiment, stimFinger=(1, 3), prestim=1, poststim=2, threshold=.03, fsample=500):
        self.experiment = experiment
        self.prestim = prestim
        self.poststim = poststim
        self.threshold = threshold
        self.fsample = fsample
        self.stimFinger = stimFinger

        self.clamped_f, self.latency = self._process_clamped()

    def _process_clamped(self):
        clamped = load_npy(self.experiment, 'clamped', 'mov')
        d = load_dat(self.experiment, 'clamped')
        Z = indicator(d.stimFinger).astype(bool)
        n_stimF = Z.shape[1]
        clamped_f = list()
        latency = list()
        for sf in range(n_stimF):
            c_mean = clamped[Z[:, sf], self.stimFinger[sf]].mean(axis=0)
            clamped_f.append(c_mean)
            latency.append(detect_response_latency(c_mean, threshold=self.threshold,
                                                   fsample=self.fsample) - self.prestim)

        return clamped_f, latency

    def timeAx(self):

        tAx = list()
        for sf in range(len(self.stimFinger)):
            tAx.append(np.linspace(-self.prestim, self.poststim,
                           int((self.prestim + self.poststim) * self.fsample)) - self.latency[sf])

        return tAx
