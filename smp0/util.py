import numpy as np
import matplotlib.pyplot as plt
# import matplotlib

# matplotlib.use('MacOSX')  # Use a non-interactive backend

from smp0.load_data import load_mov, load_dat, count_blocks


def merge_blocks_mov(experiment, participant_id):
    rawF = []
    vizF = []
    t = []

    for block in range(count_blocks(experiment, participant_id)):

        blk = "{:02d}".format(block + 1)

        print('loading participant ' + participant_id + ' - block ' + blk)

        rawForce, vizForce, time = load_mov(experiment, participant_id, blk)
        num_of_trials = len(time)

        for ntrial in range(num_of_trials):
            rawF.append(rawForce[ntrial])
            vizF.append(vizForce[ntrial])
            t.append(time[ntrial])

    return rawF, vizF, t


def align_force_to_stim(force, time, num_chan=5, pre_stim_time=1, post_stim_time=2, fsample=500):
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


def sort_by_probability(experiment, participant_id, stimFinger, datatype='raw'):
    """

    :type stimFinger: int
    :param experiment: experiment you are analyzing (e.g., smp0, smp1)
    :param participant_id: 100, 101, ...
    :param stimFinger: stimulated finger: 91999 (index), 99919 (ring)
    :param datatype: raw (raw force measured with sensors, 5 channels) or viz (visualized force on screen, 2 channels)
    :return: data sorted by probability assigned to the stimulated finger
    """

    dat = load_dat(experiment, participant_id)
    indices = dat.index[dat.stimFinger == stimFinger]
    dat = dat.loc[indices]

    match datatype:
        case 'raw':
            force, _, time = merge_blocks_mov(experiment, participant_id)
        case 'viz':
            _, force, time = merge_blocks_mov(experiment, participant_id)
        case _:
            force, time = None, None

    force = force[indices.to_numpy()]
    time = time[indices]

    aligned_force, tAx = align_force_to_stim(force, time)

    probCues = [[], [], [], [], []]

    for c, chordID in enumerate(dat.chordID):

        match chordID:
            case '39':
                probCues[0].append(aligned_force[c])
            case '93':
                probCues[1].append(aligned_force[c])
            case '12':
                probCues[2].append(aligned_force[c])
            case '21':
                probCues[3].append(aligned_force[c])
            case '44':
                probCues[4].append(aligned_force[c])

    return probCues, tAx


# def average_response_finger(experiment, participant_id, block, finger, datatype='raw', num_chan=5, plot=True, out=True):
#     # finger: value from 1 to 5
#     finger = finger - 1
#
#     fstim = [19999, 91999, 99199, 99919, 99991]
#     stim = fstim[finger]
#
#     rawForce, vizForce, time = load_mov(experiment, participant_id, block)
#     dat = load_dat(experiment, participant_id)
#     dat_block = dat[dat.BN == int(block)]
#
#     trial_indices = dat_block.index[dat_block['stimFinger'] == stim]
#
#     if datatype == 'raw':
#         aligned_force, tAx = align_force_to_stim(rawForce, time, num_chan)
#     elif datatype == 'viz':
#         aligned_force, tAx = align_force_to_stim(vizForce, time, num_chan)
#     else:
#         raise RuntimeError('Wrong input to datatype')
#
#     mean = aligned_force[trial_indices, :, finger].mean(axis=0).squeeze().astype(np.float64)
#     sd = aligned_force[trial_indices, :, finger].std(axis=0).squeeze().astype(np.float64)
#
#     if plot == True:
#         fig, axs = plt.subplots()
#
#         # axs.plot([1, 2, 3], [4, 5, 6])
#         axs.plot(tAx, mean)
#         axs.fill_between(tAx, mean - sd, mean + sd, alpha=.2)
#
#         plt.show()
#
#     if out == True:
#         return mean, sd
