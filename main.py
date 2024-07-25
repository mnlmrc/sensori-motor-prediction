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

                print(f"Saving participant {p}, session {session}...")
                np.savez(os.path.join(force.get_path(), f'{force.experiment}_{force.sn}.npz'),
                         data_array=force_segmented, descriptor=descr, allow_pickle=False)

        case 'FORCE:timec_avg':

            force_avg = list()
            for p in participant_id:
                force = Force(experiment, session, p)
                force_avg.append(force.calc_avg_timec())

            force_avg = np.array(force_avg)

            return force_avg

        case 'FORCE:timec2bins':

            df = pd.DataFrame()
            for p in participant_id:
                force = Force(experiment, session, p)
                df = pd.concat([df, force.calc_bins()])

            df.to_csv(os.path.join(gl.baseDir, experiment, session, 'bins.force.csv'))

            return df

        case 'FORCE:dist_timec':

            for p in participant_id:
                force = Force(experiment, session, p)
                dist, descr = force.calc_dist_timec(method='crossnobis')

                print(f"Saving participant {p}, session {session}...")
                np.savez(os.path.join(force.get_path(), f'{force.experiment}_{force.sn}_dist.npz'),
                         data_array=dist, descriptor=descr, allow_pickle=False)

        case 'PLOT:bins_force':

            pass

        case 'PLOT:timec_force':

            force = main('FORCE:timec_avg')
            clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

            fig, axs = plot_timec(force,
                                  channels=gl.channels['mov'],
                                  clamp=clamp,
                                  xlim=[-.1, .5],
                                  ylim=[0, 40],
                                  title=session)

            fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'force.timec.behav.png'))

            plt.show()

        case 'PLOT:timec_dist_force':

            dist = list()
            for p in participant_id:
                path = Force(experiment, session, p).get_path()
                sn = int(''.join([c for c in p if c.isdigit()]))
                npz = np.load(os.path.join(path, f'{experiment}_{sn}_dist.npz'))
                dist.append(npz['data_array'])

            dist = np.array(dist)
            y = dist.mean(axis=0)
            yerr = dist.std(axis=0) / np.sqrt(len(participant_id))

            latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t').mean(axis=1).to_numpy()

            fig, axs = plt.subplots()
            color = ['darkorange', 'darkviolet']
            label = ['stimFinger', 'cue']
            tAx = np.linspace(-gl.prestim, gl.poststim, int(gl.fsample_mov * (gl.poststim + gl.prestim))) - latency

            for i in range(2):
                axs.plot(tAx, y[i], color=color[i], label=label[i])
                axs.plot(tAx, dist[:, i].T, color=color[i], alpha=0.2)

            axs.axvline(x=0, ls='-', color='k', lw=.8)
            axs.axhline(y=0, ls='-', color='k', lw=.8)

            axs.set_yscale('symlog', linthresh=.1)
            axs.set_ylim([-.1, 200])
            axs.set_xlim([-.3, .5])

            axs.set_title(f'{session}')

            axs.set_xlabel('time relative to perturbation (s)')
            axs.set_ylabel('cross-validated multivariate distance (a.u.)')
            axs.legend()

            plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', default='PLOT:timec2bins')
    parser.add_argument('--experiment', default='smp2', help='')
    parser.add_argument('--session', default='pilot', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    session = args.session
    participant_id = args.participant_id

    if participant_id is None:
        participant_id = gl.participants[experiment]

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    main(what)
