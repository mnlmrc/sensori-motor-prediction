import json
import os
import warnings
import numpy as np
import pandas as pd
import globals as gl

import rsatoolbox as rsa

from rsa import calc_rdm_unbalanced, calc_rdm


class Force:
    def __init__(self, experiment, session, participant_id=None):

        self.experiment = experiment
        self.session = session

        self.path = self.get_path()

        self.pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

        self.prestim = int(gl.prestim * gl.fsample_mov)
        self.poststim = int(gl.poststim * gl.fsample_mov)

        # time windows in seconds
        self.win = {
            'Pre': (-.5, 0),
            'LLR': (.1, .4),
            'VOl': (.4, 1)
        }

        if participant_id is not None:
            self.participant_id = participant_id
            self.sn = int(''.join(filter(str.isdigit, participant_id)))
            self.dat = pd.read_csv(os.path.join(self.path, participant_id, f'{experiment}_{self.sn}.dat'), sep='\t')

    def get_path(self):

        session = self.session
        experiment = self.experiment
        # participant_id = self.participant_id

        if session == 'scanning':
            path = os.path.join(gl.baseDir, experiment, gl.behavDir)
        elif session == 'training':
            path = os.path.join(gl.baseDir, experiment, gl.trainDir)
        elif session == 'behavioural':
            path = os.path.join(gl.baseDir, experiment, gl.behavDir)
        elif session == 'pilot':
            path = os.path.join(gl.baseDir, experiment, gl.pilotDir)
        else:
            raise ValueError('Session name not recognized.')

        return path

    def get_block(self):

        pinfo = self.pinfo
        sn = self.sn

        if self.session == 'scanning':
            blocks = pinfo[pinfo.sn == sn].runsSess1.iloc[0].split('.')
        elif self.session == 'training':
            blocks = pinfo[pinfo.sn == sn].runsTraining.iloc[0].split('.')
        elif self.session == 'behavioural':
            blocks = pinfo[pinfo.sn == sn].blocks_mov.iloc[0].split('.')
        elif self.session == 'pilot':
            blocks = pinfo[pinfo.sn == sn].blocks_mov.iloc[0].split('.')
        else:
            raise ValueError('Session name not recognized.')

        return blocks

    def sec2sample(self, sec):
        samples = self.prestim + int(sec * gl.fsample_mov)

        return samples

    def load_mov(self, filename):
        try:
            with open(filename, 'rt') as fid:
                trial = 0
                A = []
                for line in fid:
                    if line.startswith('Trial'):
                        trial_number = int(line.split(' ')[1])
                        trial += 1
                        if trial_number != trial:
                            warnings.warn('Trials out of sequence')
                            trial = trial_number
                        A.append([])
                    else:
                        data = np.fromstring(line, sep=' ')
                        if A:
                            A[-1].append(data)
                        else:
                            warnings.warn('Data without trial heading detected')
                            A.append([data])

                mov = [np.array(trial_data) for trial_data in A]

        except IOError as e:
            raise IOError(f"Could not open {filename}") from e

        return mov

    def segment_mov(self):

        sn = self.sn
        experiment = self.experiment
        participant_id = self.participant_id
        path = self.get_path()
        blocks = self.get_block()
        ch_idx = [col in gl.channels['mov'] for col in gl.col_mov[experiment]]
        prestim = self.prestim
        poststim = self.poststim

        force = []
        for bl in blocks:
            block = f'{int(bl):02d}'
            filename = os.path.join(path, participant_id, f'{experiment}_{sn}_{block}.mov')

            mov = self.load_mov(filename)
            mov = np.concatenate(mov, axis=0)

            idx = mov[:, 1] > gl.planState[experiment]
            idxD = np.diff(idx.astype(int))
            stimOnset = np.where(idxD == 1)[0]

            print(f'Processing... {self.participant_id}, block {bl}, {len(stimOnset)} trials found...')

            for ons, onset in enumerate(stimOnset):
                # if self.dat.GoNogo.iloc[ons] == 'go':
                force.append(mov[onset - prestim:onset + poststim, ch_idx].T)

        descr = json.dumps({
            'experiment': self.experiment,
            'participant': self.participant_id,
            'fsample': gl.fsample_mov,
            'prestim': prestim,
            'poststim': poststim,
        })

        return np.array(force), descr

    def calc_avg_timec(self, GoNogo='go'):
        """
        Calculate the average force data across trials for each cue and stimulation finger.

        Returns:
            force_avg (numpy.ndarray): A 4D array with dimensions (cue, stimFinger, channel, time).
        """

        force = self.load_npz()
        blocks = self.get_block()

        # take only rows in dat that belong to good blocks based on participants.tsv
        dat = self.dat[(self.dat.BN.isin(blocks) |
                        self.dat.BN.isin(np.array(list(map(int, blocks)))))]

        keep_trials = (dat.GoNogo == GoNogo)
        force = force[keep_trials]
        dat = dat[keep_trials]

        if GoNogo == 'go':
            force_avg = np.zeros((len(gl.cue_code), len(gl.stimFinger_code), force.shape[-2], force.shape[-1]))
            for c, cue in enumerate(gl.cue_code):
                for sf, stimF in enumerate(gl.stimFinger_code):
                    force_avg[c, sf] = force[(dat.cue == cue) & (dat.stimFinger == stimF)].mean(axis=0, keepdims=True)
        elif GoNogo == 'nogo':
            force_avg = np.zeros((len(gl.cue_code), force.shape[-2], force.shape[-1]))
            for c, cue in enumerate(gl.cue_code):
                force_avg[c] = force[(dat.cue == cue)].mean(axis=0, keepdims=True)
        else:
            force_avg = None

        return force_avg

    def calc_bins(self):

        prestim = self.prestim

        force = self.load_npz()

        df = pd.DataFrame()
        for w in win.keys():
            for c, ch in enumerate(gl.channels['mov']):
                df[f'{w}/{ch}'] = force[:, c, win[w][0]:win[w][1]].mean(axis=-1)

        df = pd.concat([self.dat, df], axis=1)

        return df

    def load_npz(self):

        path = self.get_path()
        experiment = self.experiment
        participant_id = self.participant_id

        sn = int(''.join([c for c in participant_id if c.isdigit()]))

        npz = np.load(os.path.join(path, participant_id, f'{experiment}_{sn}.npz'))
        force = npz['data_array']

        return force

    def calc_rdm(self, timew, GoNogo='go'):

        force = self.load_npz()
        blocks = self.get_block()

        # take only rows in dat that belong to good blocks based on participants.tsv
        dat = self.dat[(self.dat.BN.isin(blocks) |
                        self.dat.BN.isin(np.array(list(map(int, blocks)))))]

        keep_trials = (dat.GoNogo == GoNogo)
        force = force[keep_trials]
        dat = dat[keep_trials]

        run = dat.BN
        cue = dat.cue
        stimFinger = dat.stimFinger

        cue = cue.map(gl.cue_mapping)
        stimFinger = stimFinger.map(gl.stimFinger_mapping)

        cond_vec = [f'{sf},{c}' for c, sf in zip(cue, stimFinger)]

        timew = (self.sec2sample(timew[0]), self.sec2sample(timew[1]))
        timew = np.arange(timew[0], timew[1])

        rdm = calc_rdm_unbalanced(force[..., timew].mean(axis=-1), gl.channels['mov'], cond_vec, run,
                                  method='crossnobis')

        if GoNogo == 'go':
            rdm.reorder(np.array([1, 2, 3, 0, 4, 5, 6, 7]))
        elif GoNogo == 'nogo':
            rdm.reorder(np.array([0, 2, 3, 4, 1]))

        return rdm

    def calc_dist_timec(self, method='euclidean', GoNogo='go'):

        force = self.load_npz()
        blocks = self.get_block()

        # take only rows in dat that belong to good blocks based on participants.tsv
        dat = self.dat[(self.dat.BN.isin(blocks) |
                        self.dat.BN.isin(np.array(list(map(int, blocks)))))]

        keep_trials = (dat.GoNogo == GoNogo)
        force = force[keep_trials]
        dat = dat[keep_trials]

        run = dat.BN
        cue = dat.cue
        stimFinger = dat.stimFinger

        cue = cue.map(gl.cue_mapping)
        stimFinger = stimFinger.map(gl.stimFinger_mapping)

        cond_vec = [f'{sf},{c}' for c, sf in zip(cue, stimFinger)]

        dist_stimFinger = np.zeros(force.shape[-1])
        dist_cue = np.zeros(force.shape[-1])
        dist_stimFinger_cue = np.zeros(force.shape[-1])
        for t in range(force.shape[-1]):
            print('participant %s' % self.participant_id + ', time point %f' % t)

            # calculate rdm for timepoint t
            force_tmp = force[:, :, t]

            rdm = calc_rdm(force_tmp, gl.channels['mov'], cond_vec, run, method=method)

            if GoNogo == 'go':
                rdm.reorder(np.array([0, 3, 1, 2, 7, 4, 6, 5]))
            elif GoNogo == 'nogo':
                rdm.reorder(np.array([1, 4, 2, 0, 3]))

            if GoNogo == 'go':
                dist_stimFinger[t] = rdm.dissimilarities[:, gl.mask_stimFinger].mean()
                dist_cue[t] = rdm.dissimilarities[:, gl.mask_cue].mean()
                dist_stimFinger_cue[t] = rdm.dissimilarities[:, gl.mask_stimFinger_cue].mean()

            elif GoNogo == 'nogo':
                dist_cue[t] = rdm.dissimilarities.mean()

        descr = json.dumps({
            'participant': self.participant_id,
            'mask_stimFinger': list(gl.mask_stimFinger.astype(str)) if GoNogo == 'go' else None,
            'mask_cue': list(gl.mask_cue.astype(str)) if GoNogo == 'go' else None,
            'mask_stimFinger_by_cue': list(gl.mask_stimFinger_cue.astype(str)) if GoNogo == 'go' else None,
            'factor_order': ['stimFinger', 'cue', 'stimFinger_by_cue'],
        })

        dist = np.stack([dist_stimFinger, dist_cue, dist_stimFinger_cue])

        return dist, descr


def calculate_difference(data, timewin, stim_finger, column, cue1='75%', cue2='25%'):
    filtered_data = data[(data['timewin'] == timewin) & (data['stimFinger'] == stim_finger)]
    cue1_data = filtered_data[filtered_data['cue'] == cue1]
    cue2_data = filtered_data[filtered_data['cue'] == cue2]

    mean_cue1 = cue1_data[column].mean()
    mean_cue2 = cue2_data[column].mean()

    return mean_cue1 - mean_cue2

# Example usage:
# smp = SensorimotorPrediction('experiment_name', 'session_type', 'participant_id')
# smp.save_npz()
# diff = SensorimotorPrediction.calculate_difference(data, timewin, stim_finger, column)


# # def merge_blocks_mov(experiment=None, folder=None, participant_id=None, blocks=None):
# #     """
# #
# #     :param experiment:
# #     :param participant_id:
# #     :param blocks: blocks list from field blocksForce in participants.tsv
# #     :return:
# #     """
# #
# #     rawForce, states = [], []
# #     for block in blocks:
# #
# #         print(f"loading participant: {participant_id} - block: {block}")
# #
# #         rawF, st = load_mov(experiment, folder, participant_id, block)
# #         num_of_trials = len(st)
# #
# #         for ntrial in range(num_of_trials):
# #             rawForce.append(rawF[ntrial])
# #             # vizF.append(vizForce[ntrial])
# #             states.append(st[ntrial])
# #
# #     return rawForce, states
#
#
# # def detect_state_change(states, planState=3):
# #     """
# #
# #     Args:
# #         states:
# #         planState:
# #
# #     Returns:
# #
# #     """
# #     idx = np.zeros(len(states)).astype(int)
# #     for st, state in enumerate(states):
# #         try:
# #             idx[st] = np.where(state > planState)[0][0]
# #         except:
# #             idx[st] = -1
# #
# #     return idx
#
#
# # def force_segment(rawForce, idx, prestim=None, poststim=None, fsample=None):
# #     """
# #
# #     :param rawForce:
# #     :param idx:
# #     :param prestim:
# #     :param poststim:
# #     :param fsample:
# #     :return:
# #     """
# #
# #     ntrials = len(rawForce)
# #     nfingers = rawForce[0].shape[-1]
# #     timepoints = int(fsample * (prestim + poststim))
# #
# #     force_segmented = np.zeros((ntrials, nfingers, timepoints))
# #     # NoResp = []
# #     for r, rawF in enumerate(rawForce):
# #         if idx[r] > 0:
# #             force_segmented[r] = (rawF[idx[r] - int(fsample * prestim):
# #                                        idx[r] + int(fsample * poststim)]).T
# #         else:
# #             pass
# #
# #     return force_segmented
#
# def segment(experiment, session, participant_id):
#     path = get_path_mov(experiment, session, participant_id)
#     dat = load_dat(experiment, session, participant_id)
#     blocks = get_block_mov(experiment, session, participant_id)
#
#     prestim = int(gl.prestim * gl.fsample_mov)
#     poststim = int(gl.poststim * gl.fsample_mov)
#
#     sn = int(''.join([c for c in participant_id if c.isdigit()]))
#
#     force = list()
#     trial_info = {
#         'cue': list(),
#         'stimFinger': list(),
#         'trialLabel': list()
#     }
#
#     columns = gl.col_mov[experiment]
#     ch_idx = [col in ['thumb', 'index', 'middle', 'ring', 'pinkie'] for col in columns]
#
#     for bl in blocks:
#
#         block = '%02d' % int(bl)
#         filename = os.path.join(path, f'{experiment}_{sn}_{block}.mov')
#
#         mov = load_mov(filename)
#         movC = np.concatenate(mov, axis=0)
#         mov_df = pd.DataFrame(movC, columns=columns)
#
#         bl_col = np.concatenate(
#             [np.zeros(tr.shape[0]) + int(bl) for tr in mov])  # make a column that specify block number
#         mov_df = pd.concat([pd.DataFrame(bl_col, columns=['block']), mov_df], axis='columns')
#
#         idx = (mov_df['state'] > gl.planState[experiment]).to_numpy().astype(int)
#         idxD = np.diff(idx)
#         stimOnset = np.where(idxD == 1)[0]
#
#         print(f'processing... {participant_id}, block {bl}, {len(stimOnset)} trials found...')
#
#         for st, ons in enumerate(stimOnset):
#             BN = mov_df['block'].iloc[ons]
#             TN = mov_df['trialNum'].iloc[ons]
#             GoNogo = dat[(dat.BN == BN) & (dat.TN == TN)].GoNogo.iloc[0]
#             if GoNogo == 'go':
#                 force.append(movC[ons - prestim:ons + poststim, ch_idx].swapaxes(0, 1))
#                 trial_info['cue'].append(dat[(dat.BN == BN) & (dat.TN == TN)]['cue'].iloc[0].astype(str))
#                 trial_info['stimFinger'].append(dat[(dat.BN == BN) & (dat.TN == TN)]['stimFinger'].iloc[0].astype(str))
#                 trial_info['trialLabel'].append(dat[(dat.BN == BN) & (dat.TN == TN)]['trialLabel'].iloc[0])
#
#     descr = json.dumps({
#         'experiment': experiment,
#         'participant': participant_id,
#         'fsample': gl.fsample_mov,
#         'prestim': prestim,
#         'poststim': poststim,
#         'columns': columns,
#         'trial_info': trial_info
#     })
#
#     return force, descr
#
#
# def save_npz(experiment, session, participant_id):
#     sn = int(''.join([c for c in participant_id if c.isdigit()]))
#
#     path = get_path_mov(experiment, session, participant_id)
#
#     force, descr = segment(experiment, session, participant_id)
#
#     print(f"Saving participant {participant_id}, session {session}...")
#     np.savez(os.path.join(path, f'{experiment}_{sn}.npz'),
#              data_array=force, descriptor=descr, allow_pickle=False)
#
#
# def calculate_difference(data, timewin, stim_finger, column, cue1='75%', cue2='25%'):
#     """
#
#     Args:
#         data:
#         timewin:
#         stim_finger:
#         column:
#         cue1:
#         cue2:
#
#     Returns:
#
#     """
#     # Filter the dataset for the given time window and stimFinger
#     filtered_data = data[(data['timewin'] == timewin) & (data['stimFinger'] == stim_finger)]
#
#     # Further filter for the specified cues
#     cue1_data = filtered_data[filtered_data['cue'] == cue1]
#     cue2_data = filtered_data[filtered_data['cue'] == cue2]
#
#     # Calculate the mean value of the specified column for each cue
#     mean_cue1 = cue1_data[column].mean()
#     mean_cue2 = cue2_data[column].mean()
#
#     # Compute the difference between these mean values
#     difference = mean_cue1 - mean_cue2
#
#     return difference
#
#
# def get_path_mov(experiment, session, participant_id):
#     if session == 'scanning':
#         path = os.path.join(gl.baseDir, experiment, gl.behavDir, participant_id)
#     elif session == 'training':
#         path = os.path.join(gl.baseDir, experiment, gl.trainDir, participant_id)
#     elif session == 'behav':
#         path = os.path.join(gl.baseDir, experiment, gl.behavDir, participant_id)
#     elif session == 'pilot':
#         path = os.path.join(gl.baseDir, experiment, gl.pilotDir, participant_id)
#     else:
#         raise ValueError('Session name not recognized.')
#
#     return path
#
#
# def get_block_mov(experiment, session, participant_id):
#     sn = int(''.join([c for c in participant_id if c.isdigit()]))
#     participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
#
#     if session == 'scanning':
#         blocks = participants[participants.sn == sn].runsSess1.iloc[0].split('.')
#     elif session == 'training':
#         blocks = participants[participants.sn == sn].runsTraining.iloc[0].split('.')
#     elif session == 'behav':
#         blocks = participants[participants.sn == sn].blocks_mov.iloc[0].split('.')
#     elif session == 'pilot':
#         blocks = participants[participants.sn == sn].blocks_mov.iloc[0].split('.')
#     else:
#         raise ValueError('Session name not recognized.')
#
#     return blocks
#
#
# def load_dat(experiment, session, participant_id):
#     path = get_path_mov(experiment, session, participant_id)
#     sn = int(''.join([c for c in participant_id if c.isdigit()]))
#     dat = pd.read_csv(os.path.join(path, f'{experiment}_{sn}.dat'), sep='\t')
#
#     return dat
#
#
#
#
# def load_mov(filename):
#     """
#     load .mov file of one block
#
#     :return:
#     """
#
#     try:
#         with open(filename, 'rt') as fid:
#             trial = 0
#             A = []
#             for line in fid:
#                 if line.startswith('Trial'):
#                     trial_number = int(line.split(' ')[1])
#                     trial += 1
#                     if trial_number != trial:
#                         warnings.warn('Trials out of sequence')
#                         trial = trial_number
#                     A.append([])
#                 else:
#                     # Convert line to a numpy array of floats and append to the last trial's list
#                     data = np.fromstring(line, sep=' ')
#                     if A:
#                         A[-1].append(data)
#                     else:
#                         # This handles the case where a data line appears before any 'Trial' line
#                         warnings.warn('Data without trial heading detected')
#                         A.append([data])
#
#             # Convert all sublists to numpy arrays
#             mov = [np.array(trial_data) for trial_data in A]
#             # # vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
#             # state = [np.array(trial_data) for trial_data in A]
#
#     except IOError as e:
#         raise IOError(f"Could not open {filename}") from e
#
#     return mov
