import matplotlib

from smp0.load_data import load_emg, count_blocks, path

matplotlib.use('MacOSX')

from smp0.util import experiment_dataframe

experiment = 'smp0'
participants_id = ['100', '101']

process = input('input process')

match process:

    case 'emg dataframe':

        for part in participants_id:

            nblocks = count_blocks(experiment, part, folder='emg', extension='.csv')

            for block in range(nblocks):
                df_emg = load_emg(experiment, part, block + 1)
                df_emg.to_csv(path + experiment + '/subj' + part + '/emg/' + experiment + '_' + part + '_' + str(
                    block + 1) + '.emg')

    case 'force dataframe':
        data = experiment_dataframe(experiment, participants_id)

    case _:
        print('command not recognized')





# participant_id = '100'
# # block = '01'
#
# cues = ['25%', '50%', '75%', '100%']
# timewins = ['25-50 ms', '50-200 ms', '100-500 ms']
# stimFingers = ['index', 'ring']
# allFingers = ['thumb', 'index', 'middle', 'ring', 'pinkie']
#
# probCues = (sort_by_probability(experiment, participant_id, 91999),
#             sort_by_probability(experiment, participant_id, 99919))
# probCues = subtract_baseline(probCues)
# tAx = probCues[0][1]
#
# meanTimewin = (average_within_timewin(probCues, [.025, .05], pre_stim_time=1, fsample=500),
#                average_within_timewin(probCues, [.05, .2], pre_stim_time=1, fsample=500),
#                average_within_timewin(probCues, [.1, .5], pre_stim_time=1, fsample=500))
#
# fResp_mean = np.zeros((len(meanTimewin), len(stimFingers), len(cues), len(allFingers)))
# fResp_sd = np.zeros((len(meanTimewin), len(stimFingers), len(cues), len(allFingers)))
# for timewin in range(len(meanTimewin)):
#     for finger in range(len(stimFingers)):
#         for cue in range(len(cues)):
#             fResp_mean[timewin, finger, cue, :] = meanTimewin[timewin][finger][cue].mean(axis=0)
#             fResp_sd[timewin, finger, cue, :] = meanTimewin[timewin][finger][cue].std(axis=0)
#
# fig1, axs1 = plt.subplots(4, 2, sharey=True, sharex=True, figsize=(6, 7))
#
# ylim = [-5, 30]
# xlim = [-.5, 1]
# lw = .5
#
# for cue in range(len(cues)):
#     for finger in range(len(stimFingers)):
#
#         axs1[cue, finger].plot(tAx, np.array(probCues[finger][0][cue]).mean(axis=0))
#
#         axs1[cue, finger].set_xlim(xlim)
#         axs1[cue, finger].set_ylim(ylim)
#
#         axs1[cue, finger].vlines(0, ylim[1], ylim[0], ls='-', color='k', lw=lw)
#         axs1[cue, finger].vlines(.025, ylim[1], ylim[0], ls='--', color='k', lw=lw)
#         axs1[cue, finger].vlines(.05, ylim[1], ylim[0], ls='--', color='k', lw=lw)
#         axs1[cue, finger].vlines(.2, ylim[1], ylim[0], ls='--', color='k', lw=lw)
#         axs1[cue, finger].vlines(.5, ylim[1], ylim[0], ls='--', color='k', lw=lw)
#         axs1[cue, finger].hlines(0, xlim[0], xlim[1], ls='-', color='k', lw=lw)
#
#         axs1[cue, finger].set_title(stimFingers[finger] + ', ' + cues[cue])
#
# fig1.legend(allFingers, loc='upper center', bbox_to_anchor=(.5, .975), ncol=5)
#
# fig1.supylabel('force (N)')
# fig1.supxlabel('time (s)')
# fig1.subplots_adjust(hspace=.4, right=.96, left=.1, bottom=.09)
# fig1.show()
#
# fig2, axs2 = plt.subplots(3, 2, sharex=True, sharey='row', figsize=(6, 7))
#
# xscatter = np.linspace(-.2, .2, len(allFingers))
# xoffset = np.linspace(0, len(cues) - 1, len(cues))
# barcolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
# barwidth = np.diff(xscatter).mean()
#
# for timewin in range(len(meanTimewin)):
#     for finger in range(len(stimFingers)):
#         for cue in range(len(cues)):
#             axs2[timewin, finger].bar(xscatter + xoffset[cue], fResp_mean[timewin, finger, cue, :],
#                                       color=barcolors, width=barwidth, yerr=fResp_sd[timewin, finger, cue, :])
#             axs2[timewin, finger].set_title(stimFingers[finger] + ', ' + timewins[timewin])
#             axs2[timewin, finger].set_xlim([-.5, 3.5])
#
#             if cue == 0:
#                 axs2[timewin, finger].hlines(0, axs2[timewin, finger].get_xlim()[0],
#                                              axs2[timewin, finger].get_xlim()[1], lw=lw, color='k')
# #
# # for c, timewin in enumerate(timewins):
# #     axs2[c, 0].text(1, 11, timewin)
#
# axs2[-1, 0].set_xticks(np.linspace(0, 3, 4))
# axs2[-1, 0].set_xticklabels(cues)
#
# handles = [mpatches.Patch(color=color) for color in barcolors]
# fig2.legend(handles=handles, labels=allFingers, loc='upper center', bbox_to_anchor=(.5, 1), ncol=5)
#
# fig2.supylabel('force (N)')
# fig2.supxlabel('cued probability')
# # fig2.tight_layout()
# fig2.subplots_adjust(hspace=.4, right=.96, left=.1, bottom=.085, top=.9)
# fig2.show()
