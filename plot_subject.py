import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import globals as gl
from experiment import Param
from visual import make_colors

if __name__ == "__main__":
    experiment = sys.argv[1]
    participant_id = sys.argv[2]
    session = sys.argv[3]

    # extract subject number
    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    if session == 'scanning':
        path = os.path.join(gl.baseDir, experiment, gl.behavDir, participant_id)
    elif session == 'training':
        path = os.path.join(gl.baseDir, experiment, gl.trainDir, participant_id)
    else:
        raise ValueError('Session name not recognized. Allowed session names are "scanning" and "training".')

    npz = np.load(os.path.join(path, f'{experiment}_{sn}.npz'))
    clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

    force = npz['data_array']
    descr = json.loads(npz['descriptor'].item())

    stimFinger = pd.Series(descr['trial_info']['stimFinger'])
    cue = pd.Series(descr['trial_info']['cue'])
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinkie']
    columns = pd.Series(descr['columns'])

    cues = pd.DataFrame([('0%', '93'),
                         ('25%', '12'),
                         ('50%', '44'),
                         ('75%', '21'),
                         ('100%', '39')],
                        columns=['label', 'instr'])
    colors = make_colors(5)

    fig, axs = plt.subplots(len(fingers), len(np.unique(stimFinger)),
                            sharey=True, sharex=True, figsize=(8, 10))
    for f, fi in enumerate(fingers):
        for c, cu in enumerate(cues.label.to_list()):
            for sf, stimF in enumerate(np.unique(stimFinger)):
                y = force[(stimFinger == stimF) & (cue == cues.instr.iloc[c]), :, columns == fi].mean(axis=0)
                timeAx = Param().timeAx()
                axs[f, sf].plot(timeAx, y, color=colors[c], label=cu)
                axs[f, sf].plot(timeAx, clamp[sf], color='k', ls='--', lw=.8)

                if (f == 0) & (sf == 0):
                    axs[f, sf].set_title(f'stimFinger:Index\n{fi}')
                elif (f == 0) & (sf == 1):
                    axs[f, sf].set_title(f'stimFinger:Ring\n{fi}')
                else:
                    axs[f, sf].set_title(fi)

    axs[0, 0].legend()
    axs[0, 0].set_xlim([-.1, 1])
    fig.supxlabel('time relative to stimulation (s)')
    fig.supylabel('force (N)')
    fig.suptitle(f'{participant_id}, session: {session}')

    fig.tight_layout()

    plt.show()

    # # Clamp = Clamped(experiment)
    # Params = Param(datatype)
    # Info_p = Info(experiment, [participant], datatype, ['stimFinger', 'cues'])
    # c_vec_f = Info(experiment, [participant], datatype, ['stimFinger']).cond_vec
    #
    # bs = (-1, 0)
    #
    # # define channels to plot for each datatype
    # channels = {
    #     'mov': ["thumb", "index", "middle", "ring", "pinkie"],
    #     'emg': Info_p.channels[0]
    # }
    # channels = channels[datatype]
    #
    # # define ylabel per datatype
    # ylabel = {
    #     'mov': 'force (N)',
    #     'emg': 'EMG (% of baseline)'
    # }
    # ylabel = ylabel[datatype]
    #
    # axvline = {
    #     'mov': ([0, .1, .5], ['-', ':', '-.']),
    #     'emg': ([0, .05, .1, .5], ['-', ':', '-.', ':'])
    # }
    # axvline = axvline[datatype]
    #
    # xlim = {
    #     'mov': [-.1, 1],
    #     'emg': [-.1, .3025],
    # }
    # xlim = xlim[datatype]
    #
    # ylim = {
    #     'mov': [-1, 15],
    #     'emg': [0, 25],
    # }
    #
    # cues = ['0%', '25%', '50%', '75%', '100%', 'clamped']
    # colors = make_colors(5)
    # colors.append((0.0, 0.0, 0.0))
    # ls = ['-', '-', '-', '-', '-', '--']
    # lh = [mlines.Line2D([], [], color=color, label=label, ls=ls)
    #       for label, color, ls in zip(cues, colors, ls)]
    # cues = (cues[1:5], cues[:4])
    # colors = (colors[1:5], colors[:4])
    #
    # # create list of 3D data (segmented trials)
    # Data = list()
    # for participant_id in Info_p.participants:
    #     data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
    #     if datatype == 'emg':
    #         bins = bin_traces(data, (bs,), fsample=Params.fsample,
    #                           offset=Params.prestim + Clamp.latency[0])
    #         data = data / bins
    #     Data.append(data)
    #
    # # create list of participants
    # Y = list_participants3D(Data, Info_p)
    #
    # av, sd, sem, _ = av_across_participants(channels, Y)
    #
    # timeAx = Params.timeAx()
    # # timeAx_c = (timeAx - Clamp.latency[0], timeAx - Clamp.latency[1])
    # # timeAx_clamped = Clamp.timeAx()
    #
    # stimFingers = ['Index', 'Ring']
    # n_stimF = len(stimFingers)
    # n_channels = len(channels)
    # n_labels = len(cues)
    #
    # fig, axs = plt.subplots(n_channels, n_stimF, figsize=(6.4, 9), sharex=True, sharey=True)
    # # fig.set_constrained_layout_pads(w_pad=4, h_pad=4, hspace=0.2, wspace=0.2)
    #
    # # Plotting data
    # for ch, channel in enumerate(channels):
    #     avr = av[channel].reshape((n_stimF, int(av[channel].shape[0] / n_stimF), av[channel].shape[-1]))
    #     semr = sem[channel].reshape((n_stimF, int(sem[channel].shape[0] / n_stimF), sem[channel].shape[-1]))
    #     for sF, stimF in enumerate(stimFingers):
    #         # add force clamped
    #         if datatype == 'mov':
    #             axs[ch, sF].plot(timeAx_clamped[sF], Clamp.clamped_f[sF], color='k', ls='--', lw=.8)
    #         elif datatype == 'emg':
    #             axr = axs[ch, sF].twinx()
    #             axr.plot(timeAx_clamped[sF], Clamp.clamped_f[sF], color='k', ls='--', lw=.8)
    #             axr.set_ylim(ylim['mov'])
    #             if sF == 0:
    #                 axr.spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    #                 axr.tick_params(left=False, bottom=False, right=False)
    #                 axr.set_yticklabels([])
    #             else:
    #                 axr.spines[['left', 'top', 'bottom']].set_visible(False)
    #                 axr.tick_params(left=False, bottom=False, )
    #
    #         for l, lab in enumerate(cues[sF]):
    #             axs[ch, sF].plot(timeAx_c[sF], avr[sF, l], color=colors[sF][l])
    #             axs[ch, sF].fill_between(timeAx_c[sF], avr[sF, l] - semr[sF, l], avr[sF, l] + semr[sF, l],
    #                                      color=colors[sF][l], lw=0, alpha=.2)
    #
    #             for vl in range(len(axvline[0])):
    #                 axs[ch, sF].axvline(axvline[0][vl], ls=axvline[1][vl], lw=.8, color='k')
    #
    #             axs[ch, sF].tick_params(bottom=False)
    #             axs[ch, sF].set_xlim(xlim)
    #             # axs[ch, sF].set_ylim(ylim[datatype])
    #
    #             if sF == 0:
    #                 axs[ch, sF].spines[['top', 'right', 'bottom']].set_visible(False)
    #             else:
    #                 axs[ch, sF].spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    #                 axs[ch, sF].tick_params(left=False)
    #
    # axs[0, 0].set_title('StimFinger:index')
    # axs[0, 1].set_title('StimFinger:ring')
    # axs[-1, 0].spines[['bottom', 'left']].set_visible(True)
    # axs[-1, -1].spines[['bottom']].set_visible(True)
    #
    # fig.supylabel(ylabel)
    # fig.supxlabel('Time (s)')
    # if datatype == 'emg':
    #     fig.text(0.97, 0.5, 'force (N)', va='center', ha='left', rotation='vertical', fontsize=12)
    #
    # fig.legend(handles=lh, loc='upper center', ncol=6, edgecolor='none', facecolor='whitesmoke')
    #
    # fig.tight_layout()
    # fig.subplots_adjust(top=.92, right=.9)
    #
    # for ch, channel in enumerate(channels):
    #     fig.text(.51, np.mean((axs[ch, 0].get_position().p0[1], axs[ch, 0].get_position().p1[1])),
    #              f"{f_str_latex(channel)}", va='center', ha='center', rotation=90)
    #
    # fig.canvas.manager.set_window_title(f'participant:{participant[0]}')
    #
    # plt.show()

    # fig.savefig(f'{base_dir}/smp0/figures/smp0_segment_{datatype}_{participant[0]}.svg')
    # fig.savefig(f'{base_dir}/smp0/figures/smp0_segment_{datatype}_{participant[0]}.png')
