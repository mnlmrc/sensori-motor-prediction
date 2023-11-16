import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from smp0.util import sort_by_probability, average_within_timewin, subtract_baseline

experiment = 'smp0'
participant_id = '100'
# block = '01'

cues = ['25%', '50%', '75%', '100%']
timewins = ['25-50 ms', '50-100 ms', '100-500 ms']
stimFingers = ['index', 'ring']
allFingers = ['thumb', 'index', 'middle', 'ring', 'pinkie']

probCues = (sort_by_probability(experiment, participant_id, 91999),
            sort_by_probability(experiment, participant_id, 99919))
probCues = subtract_baseline(probCues)
tAx = probCues[0][1]

meanTimewin = (average_within_timewin(probCues, [.025, .05], pre_stim_time=1, fsample=500),
               average_within_timewin(probCues, [.05, .1], pre_stim_time=1, fsample=500),
               average_within_timewin(probCues, [.1, .5], pre_stim_time=1, fsample=500))

fResp = np.zeros((len(meanTimewin), len(stimFingers), len(cues), len(allFingers)))
for timewin in range(len(meanTimewin)):
    for finger in range(len(stimFingers)):
        for cue in range(len(cues)):
            fResp[timewin, finger, cue, :] = meanTimewin[timewin][finger][cue].mean(axis=0)

fig1, axs1 = plt.subplots(4, 2, sharey=True, sharex=True)

ylim = [-5, 20]
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

fig1.legend(allFingers, loc='upper center', bbox_to_anchor=(.5, .975), ncol=5)

fig1.supylabel('force (N)')
fig1.supxlabel('time (s)')

fig1.show()

fig2, axs2 = plt.subplots(3, 2, sharex=True, sharey=True)

xscatter = np.linspace(-.2, .2, len(allFingers))
xoffset = np.linspace(1, len(cues), len(cues))
barcolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
barwidth = np.diff(xscatter).mean() / 2

for timewin in range(len(meanTimewin)):
    for finger in range(len(stimFingers)):
        for cue in range(len(cues)):
            axs2[timewin, finger].bar(xscatter + xoffset[cue], fResp[timewin, finger, cue, :],
                              color=barcolors, width=barwidth)
for c, timewin in enumerate(timewins):
    axs2[c, 0].text(1, 11, timewin)

handles = [mpatches.Patch(color=color) for color in barcolors]
fig2.legend(handles=handles, labels=allFingers, loc='upper center', bbox_to_anchor=(.5, .975), ncol=5)

fig2.supylabel('force (N)')
fig2.supxlabel('probability')

fig2.show()


