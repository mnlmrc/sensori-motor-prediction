import numpy as np

from smp0.util import merge_blocks_mov, sort_by_probability
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

fig, axs = plt.subplots(4, 2)

for row in range(len(cues)):
    for col in range(len(stimFingers)):

        axs[row, col].plot(tAx, np.array(probCues[col][0][row]).mean(axis=0))

axs[0, 0].legend(allFingers)

fig.show()


