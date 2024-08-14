import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

import globals as gl

import pandas as pd

from force import Force
from plot import make_colors, make_tAx, make_yref

import tkinter as tk

import seaborn as sns

from rsa import plot_rdm, draw_contours


def GUI():
    def on_submit():
        global what, experiment, session, participant_id
        what = case_var.get()
        experiment = experiment_entry.get() or None
        session = session_entry.get() or None
        participant_id = participant_id_entry.get().split(',') if participant_id_entry.get() else None
        root.destroy()

    root = tk.Tk()
    root.title("Case Selector")

    tk.Label(root, text="Select Case:").pack()
    case_var = tk.StringVar(root)
    case_var.set("FORCE:mov2npz")  # default value
    # cases = [
    #     "FORCE:mov2npz",
    #     "FORCE:timec_avg",
    #     "FORCE:timec2bins",
    #     "FORCE:dist_timec",
    #     "PLOT:bins_force",
    #     "PLOT:timec_force",
    #     "PLOT:timec_dist_force",
    # ]
    case_menu = tk.OptionMenu(root, case_var, *cases)
    case_menu.pack()

    tk.Label(root, text="Experiment:").pack()
    experiment_entry = tk.Entry(root)
    experiment_entry.insert(0, "smp<n>")  # default value
    experiment_entry.pack()

    tk.Label(root, text="Session:").pack()
    session_entry = tk.Entry(root)
    session_entry.insert(0, "<session>")  # default value
    session_entry.pack()

    tk.Label(root, text="Participant ID (comma-separated):").pack()
    participant_id_entry = tk.Entry(root)
    participant_id_entry.pack()

    submit_button = tk.Button(root, text="Run", command=on_submit)
    submit_button.pack()

    root.mainloop()


def main(what, experiment=None, session=None, participant_id=None, varargin=None):

    if varargin is None:
        varargin = {}

    match what:
        case 'FORCE:mov2npz':
            for p in participant_id:
                force = Force(experiment, session, p)
                force_segmented, descr = force.segment_mov()

                print(f"Saving participant {p}, session {session}...")
                np.savez(os.path.join(force.get_path(), p, f'{force.experiment}_{force.sn}.npz'),
                         data_array=force_segmented, descriptor=descr, allow_pickle=False)

        case 'FORCE:timec_avg':

            GoNogo = varargin['GoNogo'] if 'GoNogo' in varargin else 'go'

            force_avg = list()
            for p in participant_id:
                force = Force(experiment, session, p)
                force_avg.append(force.calc_avg_timec(GoNogo=GoNogo))

            force_avg = np.array(force_avg)

            return force_avg

        case 'FORCE:timec2bins':

            df = pd.DataFrame()
            for p in participant_id:
                force = Force(experiment, session, p)
                df_tmp = force.calc_bins()
                df_tmp['participant_id'] = p
                df = pd.concat([df, df_tmp])

            df.to_csv(os.path.join(Force(experiment, session).get_path(), 'bins.force.csv'))

            return df

        case 'FORCE:timec_dist':

            GoNogo = varargin['GoNogo'] if 'GoNogo' in varargin else 'go'

            for p in participant_id:
                force = Force(experiment, session, p)
                dist, descr = force.calc_dist_timec(method='crossnobis', GoNogo=GoNogo)

                print(f"Saving participant {p}, session {session}...")
                np.savez(os.path.join(force.get_path(), p, f'{force.experiment}_{force.sn}_dist_{GoNogo}.npz'),
                         data_array=dist, descriptor=descr, allow_pickle=False)

        case 'PLOT:timec_force':

            GoNogo = varargin['GoNogo'] if 'GoNogo' in varargin else 'go'
            vsep = float(varargin['vsep']) if 'vsep' in varargin else 8
            xlim = varargin['xlim'] if 'xlim' in varargin else [-.1, .5]
            ylim = varargin['ylim'] if 'ylim' in varargin else None
            title = varargin['title'] if 'title' in varargin else f'{session}, N={len(participant_id)}'
            ref_len = float(varargin['ref_len']) if 'ref_len' in varargin else 5

            force = main('FORCE:timec_avg', experiment, session, participant_id, varargin)
            # clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

            tAx = make_tAx(force) if GoNogo == 'go' else make_tAx(force, (0, 0))

            fig, axs = plt.subplots(1, 2 if GoNogo == 'go' else 1, sharey=True, sharex=True, figsize=(4, 6))

            colors = make_colors(5)
            palette = {cue: color for cue, color in zip(gl.clabels, colors)}

            for col, color in enumerate(palette):
                for c, ch in enumerate(gl.channels['mov']):
                    if GoNogo == 'go':
                        for sf, stimF in enumerate(['index', 'ring']):
                            axs[sf].set_title(f'{stimF} perturbation')

                            y = force.mean(axis=0)[:, sf, c] + c * vsep
                            yerr = force.std(axis=0)[:, sf, c] / np.sqrt(force.shape[0])

                            axs[sf].plot(tAx[sf], y[col], color=palette[color])
                            axs[sf].fill_between(tAx[sf], y[col] - yerr[col], y[col] + yerr[col],
                                                 color=palette[color], lw=0, alpha=.2)

                    elif GoNogo == 'nogo':

                        axs.set_title(f'nogo trials')

                        y = force.mean(axis=0)[:, c] + c * vsep
                        yerr = force.std(axis=0)[:, c] / np.sqrt(force.shape[0])
                        axs.plot(tAx[0], y[col], color=palette[color])
                        axs.fill_between(tAx[0], y[col] - yerr[col], y[col] + yerr[col],
                                         color=palette[color], lw=0, alpha=.2)

            if GoNogo == 'go':

                for ax in axs:
                    ax.set_xlim(xlim)
                    ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
                    ax.axvline(0, ls='-', color='k', lw=.8)
                    ax.set_yticks([])
                    ax.set_ylim(ylim)
                    ax.spines[['bottom']].set_visible(True)

                    for c, ch in enumerate(gl.channels['mov']):
                        ax.axhline(c * vsep, ls='-', color='k', lw=.8)
                        ax.text(xlim[1], c * vsep, ch, va='top', ha='right')

                make_yref(axs[0], reference_length=5)

                for c, col in enumerate(colors):
                    axs[0].plot(np.nan, label=gl.clabels[c], color=col)

            elif GoNogo == 'nogo':

                axs.set_xlim(xlim)
                axs.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
                axs.axvline(0, ls='-', color='k', lw=.8)
                axs.set_yticks([])

                for c, ch in enumerate(gl.channels['mov']):
                    axs.axhline(c * vsep, ls='-', color='k', lw=.8)
                    axs.text(xlim[1], c * vsep, ch, va='top', ha='right')

                make_yref(axs, reference_length=ref_len)

                for c, col in enumerate(colors):
                    axs.plot(np.nan, label=gl.clabels[c], color=col)

            fig.legend(ncol=3, loc='upper center')
            fig.supxlabel('time relative to perturbation (s)')
            fig.suptitle(title, y=.9)
            fig.subplots_adjust(top=.82)
            # fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'force.timec.behav.png'))

            plt.show()

        case 'PLOT:bins_force':

            GoNogo = varargin['GoNogo'] if 'GoNogo' in varargin else 'go'
            timew = varargin['timew'] if 'timew' in varargin else ['Pre', 'LLR', 'Vol']
            title = varargin['title'] if 'title' in varargin else f'{session}, N={len(participant_id)}'
            yscale = varargin['yscale'] if 'yscale' in varargin else 'linear'
            ylabel = varargin['ylabel'] if 'ylabel' in varargin else 'force (N)'

            df = pd.read_csv(os.path.join(Force(experiment, session).get_path(), 'bins.force.csv'))
            df = df[df['GoNogo'] == GoNogo]

            value_vars = [f'{time}/{ch}' for time in timew for ch in gl.channels['mov']]

            df = df.groupby(['participant_id', 'cue', 'stimFinger'])[value_vars].mean().reset_index()

            df = df.melt(id_vars=['cue', 'stimFinger', 'participant_id'],
                         value_vars=value_vars,
                         var_name='timew/finger',
                         value_name='force (N)')
            df[['timew', 'finger']] = df['timew/finger'].str.split('/', expand=True)
            df = df.drop(columns=['timew/finger'])

            df['cue'] = df['cue'].map(gl.cue_mapping)

            fig, axs = plt.subplots(len(gl.channels['mov']), len(gl.stimFinger_code),
                                    figsize=(5, 8), sharex=True, sharey=True)

            colors = make_colors(5)
            palette = {cue: color for cue, color in zip(gl.clabels, colors)}

            for c, ch in enumerate(gl.channels['mov']):
                for sF, stimF in enumerate(gl.stimFinger_code):
                    df_tmp = df[(df['finger'] == ch) & (df['stimFinger'] == stimF) & df['timew'].isin(timew)]

                    sns.boxplot(data=df_tmp, ax=axs[c, sF], x='timew', y='force (N)',
                                hue='cue', hue_order=gl.clabels, palette=palette)
                    axs[c, sF].legend_.remove()  # Remove the individual legend
                    axs[c, sF].set_title(ch)

                    axs[c, sF].set_ylabel('')
                    axs[c, sF].set_xlabel('')

                    axs[c, sF].set_yscale(yscale)

            # Create a single legend at the top
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=3)

            fig.suptitle(title)
            fig.supylabel(ylabel)
            fig.suptitle(title, y=.92)
            fig.subplots_adjust(top=.85, hspace=.4, bottom=.05)

            plt.show()

        case 'PLOT:timec_dist_force':

            GoNogo = varargin['GoNogo'] if 'GoNogo' in varargin else 'go'

            dist = list()
            for p in participant_id:
                path = Force(experiment, session, p).get_path()
                sn = int(''.join([c for c in p if c.isdigit()]))
                npz = np.load(os.path.join(path, p, f'{experiment}_{sn}_dist_{GoNogo}.npz'))
                dist.append(npz['data_array'])

            dist = np.array(dist)
            y = dist.mean(axis=0)
            yerr = dist.std(axis=0) / np.sqrt(len(participant_id))

            latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'),
                                  sep='\t').mean(axis=1).to_numpy()

            fig, axs = plt.subplots()
            color = ['darkorange', 'darkviolet']
            label = ['stimFinger', 'cue']
            tAx = np.linspace(-gl.prestim, gl.poststim, int(gl.fsample_mov * (gl.poststim + gl.prestim))) - latency

            for i in range(2):
                axs.plot(tAx, y[i], color=color[i], label=label[i])
                axs.fill_between(tAx, y[i] - yerr[i], y[i] + yerr[i], color=color[i], alpha=0.2, lw=0)
                # axs.plot(tAx, dist[:, i].T, color=color[i], alpha=0.2)

            axs.axvline(x=0, ls='-', color='k', lw=.8)
            axs.axhline(y=0, ls='-', color='k', lw=.8)

            axs.set_yscale('symlog', linthresh=.1)
            axs.set_ylim([-.1, 200])
            axs.set_xlim([-.3, .5])

            axs.set_title(f'expectation effect, {session}, N={dist.shape[0]}')

            axs.set_xlabel('time relative to perturbation (s)')
            axs.set_ylabel('cross-validated multivariate distance (a.u.)')
            axs.legend()

            fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'force.timec.dist.force.png'))

            plt.show()

        case 'PLOT:rdm_force':

            timew = varargin['timew'] if 'timew' in varargin else [(-.5, 0), (.1, .4), (.4, 1)]
            GoNogo = varargin['GoNogo'] if 'GoNogo' in varargin else 'go'
            vmin = varargin['vmin'] if 'vmin' in varargin else 0
            vmax = varargin['vmax'] if 'vmax' in varargin else 35

            colors = ['purple', 'darkorange', 'darkgreen']
            symmetry = [1, 1, -1]
            masks = [gl.mask_stimFinger, gl.mask_cue]

            fig, axs = plt.subplots(1, len(timew), sharey=True, sharex=True, figsize=(10, 5))

            for T, t in enumerate(timew):
                rdm = list()
                for p in participant_id:
                    force = Force(experiment, session, p)
                    rdm.append(force.calc_rdm(t, GoNogo=GoNogo))

                cax, axs[T] = plot_rdm(rdm, ax=axs[T], vmin=vmin, vmax=vmax)
                axs[T].set_title(f'{t[0]} to {t[1]}s')

                if GoNogo == 'go':
                    draw_contours(masks, symmetry, colors, axs=axs[T])

            cbar = fig.colorbar(cax, ax=axs, orientation='horizontal', fraction=.02)
            cbar.set_label('cross-validated multivariate distance (a.u.)')

            fig.subplots_adjust(bottom=.35)
            fig.suptitle(f'{session}, {GoNogo} trials (N={len(participant_id)})')

            plt.show()

            return fig, axs

        case 'PLOT:timec_dist_force_session':

            GoNogo = varargin['GoNogo'] if 'GoNogo' in varargin else 'go'
            title = varargin['title'] if 'title' in varargin else f'expectation effect, {GoNogo} trials'

            if GoNogo == 'go':
                sessions = ['behavioural', 'training', 'scanning', 'pilot']
            elif GoNogo == 'nogo':
                sessions = ['training', 'scanning', 'pilot']

            dist = list()
            for S, s in enumerate(sessions):

                if s == 'behavioural':
                    exp = 'smp0'
                elif (s == 'training') or (s == 'scanning'):
                    exp = 'smp1'
                elif s == 'pilot':
                    exp = 'smp2'

                participants = gl.participants[exp]

                dist_tmp = list()
                for P, p in enumerate(participants):
                    path = Force(exp, s, p).get_path()
                    sn = int(''.join([c for c in p if c.isdigit()]))
                    npz = np.load(os.path.join(path, p, f'{exp}_{sn}_dist_{GoNogo}.npz'))

                    dist_tmp.append(npz['data_array'][1])

                dist.append(np.array(dist_tmp))

            fig, axs = plt.subplots()

            palette = sns.color_palette("husl", len(sessions))

            latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'),
                                  sep='\t').mean(axis=1).to_numpy()

            tAx = np.linspace(-gl.prestim, gl.poststim, int(gl.fsample_mov * (gl.poststim + gl.prestim))) - latency

            for i in range(len(sessions)):
                axs.plot(tAx, dist[i].mean(axis=0), color=palette[i], label=sessions[i])
                axs.fill_between(tAx, dist[i].mean(axis=0) - dist[i].std(axis=0) / np.sqrt(dist[i].shape[0]),
                                 dist[i].mean(axis=0) + dist[i].std(axis=0) / np.sqrt(dist[i].shape[0]),
                                 color=palette[i], alpha=0.3, lw=0)

            axs.set_ylabel('cross-validated multivariate distance (a.u.)')
            axs.set_xlabel('time relative to perturbation (s)')

            axs.axvline(x=0, ls='-', color='k', lw=.8)
            axs.axhline(y=0, ls='-', color='k', lw=.8)

            axs.set_yscale('symlog', linthresh=.1)
            axs.set_ylim([-.1, 200])
            axs.set_xlim([-.3, .5])

            axs.legend(loc='upper left')
            axs.set_title(title)

            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    cases = [
        'FORCE:mov2npz',
        'FORCE:timec_avg',
        'FORCE:timec2bins',
        'FORCE:timec_dist',
        'PLOT:bins_force',
        'PLOT:rdm_force',
        'PLOT:timec_force',
        'PLOT:timec_dist_force',
        'PLOT:timec_dist_force_session'
    ]

    parser.add_argument('what', nargs='?', default=None, choices=cases)
    parser.add_argument('--experiment', default='smp2', help='')
    parser.add_argument('--session', default='pilot', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')

    args, extra_args = parser.parse_known_args()

    what = args.what
    experiment = args.experiment
    session = args.session
    participant_id = args.participant_id

    varargin = dict()
    for i in range(0, len(extra_args), 2):
        varargin[extra_args[i][2:]] = extra_args[i + 1]

    if what is None:
        GUI()

    if participant_id is None:
        participant_id = gl.participants[experiment]

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    main(what, experiment, session, participant_id, varargin=varargin)
