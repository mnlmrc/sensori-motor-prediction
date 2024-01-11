import numpy as np
from PcmPy import indicator

from smp0.dataset import Dataset3D
from smp0.experiment import Param
from smp0.fetch import load_npy, load_dat
from smp0.utils import average_condition, detect_response_latency


def create_participants_list3D(Data, Info_p):
    """
    Creates a list of instances of :class:`smp0.dataset.Dataset3D`.

    :param Data: List of 3D numpy arrays shaped (n_trials, n_channels, n_timepoints).
    :param Info_p: Instance of :class:`smp0.experiment.Info`.
    :return: List of instances of :class:`smp0.dataset.Dataset3D`.
    """

    Y = list()
    for p, participant_id in enumerate(Info_p.participants):
        data = Data[p]
        obs_des = {'n_trials': Info_p.n_trials[p],
                   'cond_vec': Info_p.cond_vec[p]}
        ch_des = {'n_channels': len(Info_p.channels[p]),
                  'channels': Info_p.channels[p]}
        Y.append(Dataset3D(measurements=data, obs_descriptors=obs_des, channel_descriptors=ch_des))

    return Y


def create_channels_dictionary(Y, channels=None):
    """

    :param Y:
    :param channels:
    :return:
    """

    channels_dict = {ch: [] for ch in channels}
    N = len(Y)
    for p in range(N):
        Z = indicator(Y[p].obs_descriptors['cond_vec']).astype(bool)
        M = average_condition(Y[p], Z)
        for ch in channels:
            if ch in Y[p].channel_descriptors['channels']:
                channels_dict[ch].append(M[:, Y[p].channel_descriptors['channels'].index(ch)])

    Mean = {ch: np.array(channels_dict[ch]).mean(axis=0) for ch in channels}
    SD = {ch: np.array(channels_dict[ch]).std(axis=0) for ch in channels}
    SE = {ch: np.array(channels_dict[ch]).std(axis=0) / np.sqrt(N) for ch in channels}

    return Mean, SD, SE, channels_dict


def process_clamped(experiment):
    clamped = load_npy(experiment, 'clamped', 'mov')
    clamped_d = load_dat(experiment, 'clamped')
    clamped_Z = indicator(clamped_d.stimFinger).astype(bool)
    clamped_i = clamped[clamped_Z[:, 0]].mean(axis=0)
    clamped_r = clamped[clamped_Z[:, 1]].mean(axis=0)
    clamped_mean = np.stack([clamped_i, clamped_r], axis=0)
    clamped_latency = (detect_response_latency(clamped_mean[0, 1],
                                               threshold=.03, fsample=Param('mov').fsample) - Param().prestim,
                       detect_response_latency(clamped_mean[1, 3],
                                               threshold=.03, fsample=Param('mov').fsample) - Param().prestim)

    return clamped_mean, clamped_latency

# better make a class for clamped
