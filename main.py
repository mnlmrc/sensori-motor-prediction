import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

import globals as gl

import pandas as pd

from force import Force
from plot import plot_timec


def main(what):
    match what:
        case 'FORCE:mov2npz':

            for p in participant_id:
                force = Force(experiment, session, p)
                force_segmented, descr = force.segment()

        case 'FORCE:timec_avg':

            force_avg = list()
            for p in participant_id:
                force = Force(experiment, session, p)
                force_avg.append(force.calc_avg_timec())

            force_avg = np.array(force_avg)

            return force_avg

        case 'PLOT:timec_force':

            force = main('FORCE:timec_avg')
            clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

            fig, axs = plot_timec(force,
                                  channels=gl.channels['mov'],
                                  clamp=clamp,
                                  xlim=[-.1, .5],
                                  ylim=[0, 40])

            fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'force.timec.behav.png'))

            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', default='PLOT:timec_avg')
    parser.add_argument('--experiment', default='smp2', help='')
    parser.add_argument('--session', default='pilot', help='')
    parser.add_argument('--participant_id', nargs='+', default='subj100', help='')

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    session = args.session
    participant_id = args.participant_id

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    main(what)
