import numpy as np
from smp0.fetch import load_participants, load_dat
from smp0.utils import remap_chordID


class Info:

    def __init__(self, experiment, participants, datatype=None, condition_headers=None, demographics=None):

        self.experiment = experiment
        self._info = load_participants(self.experiment)  # load info from participants.tsv
        self.participants = participants

        if datatype is not None:
            self.datatype = datatype
            self.condition_headers = condition_headers
            self.cond_vec, self.channels, self.n_trials = self._process_participant_info()

        if demographics is not None:
            self.dem_info = demographics
            self.demographics = self._process_participant_demographics()

    # def _load_data_dataype(self):
    #     """
    #     Loads and processes data for each datatype.
    #
    #     Returns:
    #     dict: A nested dictionary containing processed data for each datatype.
    #     """
    #     c_vector_dict = {}
    #     channels_dict = {}
    #     n_trials = {}
    #     cond_vec, channels, n_trials = self._process_datatype(datatype)
    #     return cond_vec, channels, n_trials

    # def _process_datatype(self, datatype):
    #     """
    #     Processes data for a specific datatype.
    #
    #     Parameters:
    #     datatype (str): The datatype to process.
    #
    #     Returns:
    #     dict: A dictionary containing processed data for the specified datatype.
    #     """
    #     c_vector_dict = {}
    #     n_trials = {}
    #     channels_dict = {}
    #     info = load_participants(self.experiment)
    #     for participant_id in self.participants:
    #         c_vector_dict[participant_id], n_trials[participant_id] = self._process_datatype_condition_vectors(info, participant_id, datatype)
    #         channels_dict[participant_id] = info.at[f"subj{participant_id}", f"channels_{datatype}"].split(",")
    #     return c_vector_dict, channels_dict, n_trials

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

        # loop through participants
        for p, participant_id in enumerate(self.participants):

            # load .dat
            d = remap_chordID(load_dat(self.experiment, participant_id))

            # filter .dat using blocks available per datatype
            blocks = self._info.at[f"subj{participant_id}", f"blocks_{self.datatype}"].split(",")
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
            channels.append(self._info.at[f"subj{participant_id}", f"channels_{self.datatype}"].split(","))

        return cond_vec, channels, n_trials

    # def _process_column(self, column_data):
    #     """
    #     Processes a single column of data.
    #
    #     Parameters:
    #     column_data (str or other): The data in a column for a participant.
    #
    #     Returns:
    #     list or original data: A list if the data contains comma-separated values, otherwise the original data.
    #     """
    #     if isinstance(column_data, str) and "," in column_data:
    #         return column_data.split(",")
    #     else:
    #         return column_data

    # def load_demographics(self):
    #
    #     demographic_dict = {}
    #     for participant_id in self.participants:
    #         demographic_dict[participant_id] = self._process_participant_demographics(self._info, participant_id)
    #
    #     return demographic_dict


class Param:

    def __init__(self, datatype=None, prestim=1, poststim=2):
        self._fsample = {'emg': 2148.1481, 'mov': 500}
        if datatype is not None:
            self.fsample = self._fsample[datatype]
            self.datatype = datatype
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


# Example usage
# Info = Info(
#     experiment='smp0',
#     participants=['100', '101', '102', '103', '104', '105', '106', '107', '108', '110'],
#     datatype='emg',
#     condition_headers=['stimFinger', 'cues']
# )
#
# ExpParam = Param(
#     datatype='emg',
#     prestim=1,
#     poststim=2
#
# )
