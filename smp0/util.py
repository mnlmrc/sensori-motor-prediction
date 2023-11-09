import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')  # Use a non-interactive backend

from smp0.load_data import load_mov, load_dat

def align_force_to_stim(force, time, num_chan=5):
    pre_stim_time = 1  # minimum planTime (s)
    post_stim_time = 2  # time after stimulus in aligned data (s)
    fsample = 500  # sampling frequency (Hz)
    num_of_trials = len(force)

    aligned_force = np.zeros((num_of_trials, fsample * (pre_stim_time + post_stim_time),
                              num_chan))
    for ntrial in range(num_of_trials):
        stim_idx = np.where(time[ntrial][:, 0] > 2)[0][0]
        aligned_force[ntrial] = force[ntrial][stim_idx - fsample * pre_stim_time:
                                                     stim_idx + fsample * post_stim_time]

    tAx = np.linspace(-pre_stim_time, post_stim_time,
                      int(fsample * (pre_stim_time + post_stim_time)))

    return aligned_force, tAx

def average_response(experiment, participant_id, block, finger, datatype='raw', num_chan=5, plot=True, out=True):

    # finger: value from 1 to 5
    finger = finger - 1

    stim = '99999'
    stim = int(stim[:finger] + '1' + stim[finger + 1:])

    rawForce, vizForce, time = load_mov(experiment, participant_id, block)
    dat = load_dat(experiment, participant_id)

    trial = np.where(dat.stimFinger == stim)

    if datatype == 'raw':
        aligned_force, tAx = align_force_to_stim(rawForce, time, num_chan)
    elif datatype == 'viz':
        aligned_force, tAx = align_force_to_stim(vizForce, time, num_chan)
    else:
        raise RuntimeError('Wrong input to datatype')

    mean = aligned_force[trial, :, finger].mean(axis=1).squeeze().astype(np.float64)
    sd = aligned_force[trial, :, finger].std(axis=1).squeeze().astype(np.float64)

    if plot==True:

        fig, axs = plt.subplots()

        # axs.plot([1, 2, 3], [4, 5, 6])
        axs.plot(tAx, mean)
        axs.fill_between(tAx, mean - sd, mean + sd, alpha=.2)

        plt.show()

    if out==True:

        return mean, sd







