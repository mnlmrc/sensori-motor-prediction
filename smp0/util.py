import numpy as np


def align_force_to_stim(force, time, planTime, num_chan=5):
    min_planTime = 1500  # minimum planTime (ms)
    after_time = 3000  # time after stimulus in aligned data (ms)
    fsample = 500  # sampling frequency (Hz)
    num_of_trials = len(force)

    aligned_force = np.zeros((num_of_trials, fsample * (min_planTime + after_time),
                              num_chan))
    for ntrial in range(num_of_trials):
        stim_idx = np.where(time[ntrial][:, -1] >= planTime[ntrial])[0][0]
        aligned_force[num_of_trials] = force[ntrial][stim_idx - fsample * min_planTime:
                                                     stim_idx + fsample * after_time]

    return aligned_force
