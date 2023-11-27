import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import matplotlib

# matplotlib.use('MacOSX')  # Use a non-interactive backend

from smp0.load_data import load_mov, load_dat, count_blocks, path


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
    NoResp = []
    for ntrial in range(num_of_trials):
        # <<<<<<< Updated upstream
        try:
            stim_idx = np.where(time[ntrial][:, 0] > 2)[0][0]
            aligned_force[ntrial] = force[ntrial][stim_idx - fsample * pre_stim_time:
                                                  stim_idx + fsample * post_stim_time]
        except:
            NoResp.append(ntrial + 1)
    # =======
    #         print('trial: %d' % ntrial)
    #         stim_idx = np.where(time[ntrial][:, 0] > 2)[0][0]
    #         aligned_force[ntrial] = force[ntrial][stim_idx - fsample * pre_stim_time:
    #                                               stim_idx + fsample * post_stim_time]
    # >>>>>>> Stashed changes

    tAx = np.linspace(-pre_stim_time, post_stim_time,
                      int(fsample * (pre_stim_time + post_stim_time)))

    return aligned_force, tAx, NoResp


def participant_dataframe(experiment, participant_id):
    dat = load_dat(experiment, participant_id)
    # indices = dat.index[dat.stimFinger == stimFinger]
    # dat = dat.loc[indices]

    # match datatype:
    #     case 'raw':
    rawF, vizF, time = merge_blocks_mov(experiment, participant_id)
    # case 'viz':
    #     _, force, time = merge_blocks_mov(experiment, participant_id)
    # case _:
    #     force, time = None, None

    # fforce = [force[i] for i in indices if i < len(force)]
    # ftime = [time[i] for i in indices if i < len(time)]

    # <<<<<<< Updated upstream
    aligned_force_raw, tAx, NoResp = align_force_to_stim(rawF, time)
    # aligned_force_viz, tAx, NoResp = align_force_to_stim(vizF, time)
    # =======
    #     aligned_force, tAx = align_force_to_stim(fforce, ftime)
    # >>>>>>> Stashed changes

    data_list = []

    for index, (chordID, TN, stimFinger) in enumerate(zip(dat.chordID, dat.TN, dat.stimFinger)):
        if not TN in NoResp:
            data_list.append({
                'Experiment': experiment,
                'Participant': participant_id,
                'Trial': TN,
                'probCue': chordID,
                'stimFinger': stimFinger,
                'Thumb': list(aligned_force_raw[TN, :, 0]),
                'Index': list(aligned_force_raw[TN, :, 1]),
                'Middle': list(aligned_force_raw[TN, :, 2]),
                'Ring': list(aligned_force_raw[TN, :, 3]),
                'Pinkie': list(aligned_force_raw[TN, :, 4]),
                'timeX': list(tAx)
            })

    data = pd.DataFrame(data_list)

    return data


def experiment_dataframe(experiment, participants_id, path=path):
    data = pd.DataFrame()

    for id in participants_id:
        data = pd.concat(
            [data, participant_dataframe(experiment, id)]
        )

    data.to_csv(path + experiment + '/' + experiment + '.tsv')

    return data

def sort_by_finger_and_probability(data, stimFinger, probCue):

    indices = data.index[(data.stimFinger == stimFinger) &
                         (data.probCue == probCue)]
    data = data.loc[indices]


    # probCues._append({'Experiment': experiment,
    #                   'Participant': participant_id,
    #                   'Trial': TN,
    #                   'Probability Cue': chordID,
    #                   'Stimulated Finger': stimFinger,
    #                   'Raw Force': rawF[TN],
    #                   'Viz Force': vizF[TN],
    #                   'timeX': tAx})
    #
    # match chordID:
    #     case 39:
    #         probCues[3].append(aligned_force[index])
    #     case 93:
    #         probCues[3].append(aligned_force[index])
    #     case 12:
    #         if stimFinger == 91999:
    #             probCues[0].append(aligned_force[index])
    #         elif stimFinger == 99919:
    #             probCues[2].append(aligned_force[index])
    #     case 21:
    #         if stimFinger == 91999:
    #             probCues[2].append(aligned_force[index])
    #         elif stimFinger == 99919:
    #         #             probCues[0].append(aligned_force[index])
    #         #     case 44:
    #         #         probCues[1].append(aligned_force[index])
    #
    # return data


def average_within_timewin(probCues, timewin, pre_stim_time=1, fsample=500):
    meanTimewin = [[[], [], [], []], [[], [], [], []]]
    for finger in range(2):
        for cue in range(len(probCues[finger][0])):
            for ntrial in range(len(probCues[finger][0][cue])):
                pc = np.array(probCues[finger][0][cue][ntrial][
                              int((pre_stim_time - timewin[0]) * fsample):int((pre_stim_time + timewin[1]) * fsample)])
                meanTimewin[finger][cue].append(pc.mean(axis=0))

            meanTimewin[finger][cue] = np.array(meanTimewin[finger][cue])

    return meanTimewin


def subtract_baseline(probCues, experiment='smp0', baseline_type='clamped', block='01'):
    baseline, _, time = load_mov(experiment, baseline_type, block)
    baseline_dat = load_dat(experiment, baseline_type)

    indices1 = baseline_dat.index[baseline_dat.stimFinger == 91999].to_numpy()
    indices2 = baseline_dat.index[baseline_dat.stimFinger == 99919].to_numpy()

    baseline1 = [baseline[idx] for idx in indices1]
    baseline2 = [baseline[idx] for idx in indices2]

    aligned_baseline1, _, _ = align_force_to_stim(baseline1, time)
    aligned_baseline2, _, _ = align_force_to_stim(baseline2, time)

    mbaseline = aligned_baseline1.mean(axis=0), aligned_baseline2.mean(axis=0)

    for finger in range(len(probCues)):
        for cue in range(len(probCues[finger][0])):
            for ntrial in range(len(probCues[finger][0][cue])):
                probCues[finger][0][cue][ntrial] = probCues[finger][0][cue][ntrial] - mbaseline[finger]

    return probCues

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
