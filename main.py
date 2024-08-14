import argparse
import os
import nitools.gifti
import numpy as np
from matplotlib import pyplot as plt
import globals as gl
import pandas as pd
from force import Force
from plot import make_colors, make_tAx, make_yref
import tkinter as tk
import seaborn as sns
from rsa import plot_rdm, draw_contours
import nibabel as nb

import sys

sys.path.append('/Users/mnlmrc/Documents/GitHub')

import surfAnalysisPy as surf


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


def main(what, experiment=None, session=None, participant_id=None, GoNogo=None, glm=None, Hem=None, regressor=None,
         fig=None, axs=None, vsep=None, xlim=None, ylim=None, vmin=None, vmax=None, ref_len=None):
    if participant_id is None:
        participant_id = gl.participants[experiment]

    match what:
        case 'FORCE:mov2npz':
            for p in participant_id:
                force = Force(experiment, session, p)
                force_segmented, descr = force.segment_mov()

                print(f"Saving participant {p}, session {session}...")
                np.savez(os.path.join(force.get_path(), p, f'{force.experiment}_{force.sn}.npz'),
                         data_array=force_segmented, descriptor=descr, allow_pickle=False)

        case 'FORCE:timec_avg':

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

            for p in participant_id:
                force = Force(experiment, session, p)
                dist, descr = force.calc_dist_timec(method='crossnobis', GoNogo=GoNogo)

                print(f"Saving participant {p}, session {session}...")
                np.savez(os.path.join(force.get_path(), p, f'{force.experiment}_{force.sn}_dist_{GoNogo}.npz'),
                         data_array=dist, descriptor=descr, allow_pickle=False)

        case 'PLOT:timec_force':

            if fig is None or axs is None:
                fig, axs = plt.subplots(1, 2 if GoNogo == 'go' else 1, sharey=True, sharex=True, figsize=(4, 6))

            force = main('FORCE:timec_avg', experiment, session, participant_id, GoNogo=GoNogo)
            # clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

            tAx = make_tAx(force) if GoNogo == 'go' else make_tAx(force, (0, 0))

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

            # fig.legend(ncol=3, loc='upper center')
            # fig.supxlabel('time relative to perturbation (s)')
            # fig.suptitle(title, y=.9)
            # fig.subplots_adjust(top=.82)
            # fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'force.timec.behav.png'))

            return fig, axs

        case 'PLOT:bins_force':

            if fig is None or axs is None:
                fig, axs = plt.subplots(len(gl.channels['mov']), len(gl.stimFinger_code),
                                        figsize=(5, 8), sharex=True, sharey=True)

            timew = ['Pre', 'LLR', 'Vol']

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

                    # axs[c, sF].set_yscale(yscale)

            # Create a single legend at the top
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=3)

            # fig.suptitle(title)
            # fig.supylabel(ylabel)
            # fig.suptitle(title, y=.92)
            # fig.subplots_adjust(top=.85, hspace=.4, bottom=.05)

            return fig, axs

        case 'PLOT:timec_dist_force':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

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

            return fig, axs

        case 'PLOT:rdm_force':

            timew = [(-.5, 0), (.1, .4), (.4, 1)]

            if fig is None or axs is None:
                fig, axs = plt.subplots(1, len(timew), sharey=True, sharex=True, figsize=(10, 5))

            colors = ['purple', 'darkorange', 'darkgreen']
            symmetry = [1, 1, -1]
            masks = [gl.mask_stimFinger, gl.mask_cue]

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

            if fig is None or axs is None:
                fig, axs = plt.subplots()

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

            return fig, axs

        case 'PLOT:flatmap':
            if fig is None or axs is None:
                fig, axs = plt.subplots()

            darray_avg_subj = list()
            for p in participant_id:
                data = os.path.join(gl.baseDir, experiment, gl.wbDir, p, f'glm{glm}.spmT.{Hem}.func.gii')

                D = nb.load(data)
                darray = nitools.get_gifti_data_matrix(D)

                # make indeces to plot
                col_names = nitools.get_gifti_column_names(D)
                regressor = [f'spmT_{r}.nii' for r in regressor]
                im = np.array([x in regressor for x in col_names])

                # avg darray
                darray_avg_subj.append(darray[:, im].mean(axis=1))

            darray_avg = np.array(darray_avg_subj).mean(axis=0)
            plt.sca(axs)
            surf.plot.plotmap(darray_avg, f'fs32k_{Hem}',
                              underlay=None,
                              borders=gl.borders[Hem],
                              cscale=[vmin, vmax],
                              cmap='jet',
                              underscale=[-1.5, 1],
                              alpha=.5,
                              new_figure=False,
                              colorbar=False,
                              frame=[xlim[0], xlim[1], ylim[0], ylim[1]])

            return fig, axs


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
        'PLOT:timec_dist_force_session',
        'PLOT:flatmap'
    ]

    parser.add_argument('what', nargs='?', default=None, choices=cases)
    parser.add_argument('--experiment', default='smp2', help='')
    parser.add_argument('--session', default='pilot', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')
    parser.add_argument('--glm', default=None, help='')
    parser.add_argument('--Hem', default=None, help='')
    parser.add_argument('--regressor', nargs='+', default=None, help='')

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    session = args.session
    participant_id = args.participant_id
    glm = args.glm
    Hem = args.Hem
    regressor = args.regressor

    if what is None:
        GUI()

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    main(what=what, experiment=experiment, session=session, participant_id=participant_id, glm=glm, Hem=Hem,
         regressor=regressor)

    plt.show()
