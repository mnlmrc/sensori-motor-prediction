import numpy as np
from smp0.util import merge_blocks_mov, sort_by_probability, average_within_timewin
import matplotlib.pyplot as plt

experiment = 'smp0'
participant_id = '100'
# block = '01'

cues = ['25%', '50%', '75%', '100%']
stimFingers = ['index', 'ring']
allFingers = ['thumb', 'index', 'middle', 'ring', 'pinkie']

probCues = (sort_by_probability(experiment, participant_id, 91999),
            sort_by_probability(experiment, participant_id, 99919))
tAx = probCues[0][1]

fig1, axs1 = plt.subplots(4, 2, sharey=True, sharex=True)
fig2, axs2 = plt.subplots()

ylim = [0, 30]
lw = .5

for row in range(len(cues)):
    for col in range(len(stimFingers)):

        axs1[row, col].plot(tAx, np.array(probCues[col][0][row]).mean(axis=0))

        axs1[row, col].set_xlim([-.5, 1])
        axs1[row, col].set_ylim(ylim)
        axs1[row, col].vlines(0, ylim[1], 0, ls='-', color='k', lw=lw)
        axs1[row, col].vlines(.025, ylim[1], 0, ls='--', color='k', lw=lw)
        axs1[row, col].vlines(.05, ylim[1], 0, ls='--', color='k', lw=lw)
        axs1[row, col].vlines(.1, ylim[1], 0, ls='--', color='k', lw=lw)
        axs1[row, col].vlines(.5, ylim[1], 0, ls='--', color='k', lw=lw)

for cue in range(len(cues)):

    meanTimewinSLR = average_within_timewin(probCues, [.025, .5], pre_stim_time=1, fsample=500)
    meanTimewinLLR = average_within_timewin(probCues, [.05, .1], pre_stim_time=1, fsample=500)

fig1.legend(allFingers, loc='upper center', bbox_to_anchor=(.5, .975), ncol=5)

fig1.supylabel('force (N)')
fig1.supxlabel('time (s)')

fig1.show()


