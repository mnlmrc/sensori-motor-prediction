import argparse
import os
import json

from utils import moving_average

from scipy.signal import decimate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import globals as gl
import rsatoolbox as rsa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--participant_id', default='subj109', help='Participant ID')
    parser.add_argument('--method', default='crossnobis', help='Distance')
    parser.add_argument('--experiment', default='smp0', help='Experiment')
    parser.add_argument('--make_plot', default=False, help='Make plot for single subject')
    parser.add_argument('--session', default='behavioural', help='Session')

    args = parser.parse_args()

    participant_id = args.participant_id
    method = args.method
    experiment = args.experiment
    make_plot = args.make_plot
    session = args.session

    path = os.path.join(gl.baseDir, experiment)

    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    fsample = 500

    participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    if session is 'behavioural':
        channels = participants[participants['sn'] == sn].channels_mov.iloc[0].split(',')
        blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_mov.iloc[0].split('.')]
        dat = pd.read_csv(os.path.join(path, participant_id, f'{experiment}_{sn}.dat'), sep='\t')
        dat = dat[dat.BN.isin(blocks)]
        cue = dat.chordID
        stimFinger = dat.stimFinger
        run = dat.BN
        force = np.load(os.path.join(path, participant_id, 'mov', f'{experiment}_{sn}.npy'))
        out_path = os.path.join(path, participant_id, 'mov')
    # elif experiment is 'smp1':
    #     channels = ['thumb', 'index', 'middle', 'ring', 'pinkie']
    elif session is 'scanning':
        blocks = [int(b) for b in participants[participants['sn'] == sn].runsSess1.iloc[0].split('.')]
        dat = pd.read_csv(os.path.join(path, gl.behavDir, participant_id, f'{experiment}_{sn}.dat'), sep='\t')
        dat = dat[dat.BN.isin(blocks)]
        dat = dat[dat.GoNogo == 'go']
        cue = dat.cue
        stimFinger = dat.stimFinger
        run = dat.BN
        force = np.load(os.path.join(path, gl.behavDir, participant_id, f'{experiment}_{sn}.npz'))['data_array']
        out_path = os.path.join(path, gl.behavDir, participant_id)
    elif session is 'training':
        blocks = [int(b) for b in participants[participants['sn'] == sn].runsTraining.iloc[0].split('.')]
        dat = pd.read_csv(os.path.join(path, gl.trainDir, participant_id, f'{experiment}_{sn}.dat'), sep='\t')
        dat = dat[dat.BN.isin(blocks)]
        dat = dat[dat.GoNogo == 'go']
        cue = dat.cue
        stimFinger = dat.stimFinger
        run = dat.BN
        force = np.load(os.path.join(path, gl.trainDir, participant_id, f'{experiment}_{sn}.npz'))['data_array']
        out_path = os.path.join(path, gl.trainDir, participant_id)

    win_size = 10
    force = moving_average(force, win_size, axis=-1)

    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')

    map_cue = pd.DataFrame([('0%', 93),
                            ('25%', 12),
                            ('50%', 44),
                            ('75%', 21),
                            ('100%', 39)],
                           columns=['label', 'code'])
    map_dict = dict(zip(map_cue['code'], map_cue['label']))
    cue = [map_dict.get(item, item) for item in cue]

    map_stimFinger = pd.DataFrame([('index', 91999),
                                   ('ring', 99919),
                                   ('none', 99999)],
                                  columns=['label', 'code'])
    map_dict = dict(zip(map_stimFinger['code'], map_stimFinger['label']))
    stimFinger = [map_dict.get(item, item) for item in stimFinger]

    # make masks
    mask_stimFinger = np.zeros([28], dtype=bool)
    mask_cue = np.zeros([28], dtype=bool)
    mask_stimFinger_cue = np.zeros([28], dtype=bool)
    mask_stimFinger[[4, 11, 17]] = True
    mask_cue[[0, 1, 7, 25, 26, 27]] = True
    mask_stimFinger_cue[[5, 6, 10, 12, 15, 16]] = True

    # win_size = 100
    # emg = moving_average(emg, win_size, axis=-1)

    timeAx = (np.linspace(-1 + win_size / (fsample * 2), 2 - win_size / (fsample * 2), force.shape[-1]) -
              latency[['ring', 'index']].mean(axis=1).to_numpy())
    dist_stimFinger = np.zeros(force.shape[-1])
    dist_cue = np.zeros(force.shape[-1])
    dist_stimFinger_cue = np.zeros(force.shape[-1])
    for t in range(force.shape[-1]):
        print('participant %s' % participant_id + ', time point %f' % timeAx[t])

        # calculate rdm for timepoint t
        force_tmp = force[:, :, t]
        dataset = rsa.data.Dataset(
            force_tmp,
            channel_descriptors={'channels': channels},
            obs_descriptors={'stimFinger,cue': [sf + ',' + c for c, sf in zip(cue, stimFinger)], 'run': run},
        )
        noise = rsa.data.noise.prec_from_unbalanced(dataset,
                                                    obs_desc='stimFinger,cue',
                                                    method='diag')
        rdm = rsa.rdm.calc_rdm_unbalanced(dataset,
                                          method=method,
                                          descriptor='stimFinger,cue',
                                          noise=noise,
                                          cv_descriptor='run')
        rdm.reorder(rdm.pattern_descriptors['stimFinger,cue'].argsort())
        rdm.reorder(np.array([1, 2, 3, 0, 4, 5, 6, 7]))

        dist_stimFinger[t] = rdm.dissimilarities[:, mask_stimFinger].mean()
        dist_cue[t] = rdm.dissimilarities[:, mask_cue].mean()
        dist_stimFinger_cue[t] = rdm.dissimilarities[:, mask_stimFinger_cue].mean()

    descr = json.dumps({
        'participant': participant_id,
        'mask_stimFinger': list(mask_stimFinger.astype(str)),
        'mask_cue': list(mask_cue.astype(str)),
        'mask_stimFinger_by_cue': list(mask_stimFinger_cue.astype(str)),
        'factor_order': ['stimFinger', 'cue', 'stimFinger_by_cue'],
    })

    dist = np.stack([dist_stimFinger, dist_cue, dist_stimFinger_cue])
    np.savez(os.path.join(out_path, f'{experiment}_{sn}_distances.npz'),
             data_array=dist, descriptor=descr, allow_pickle=False)

    fig, axs = plt.subplots()

    axs.plot(timeAx, dist_stimFinger, label='finger')
    axs.plot(timeAx, dist_cue, label='cue')
    axs.plot(timeAx, dist_stimFinger_cue, label='interaction')
    axs.set_title(f'cross-validated distance over time, {participant_id}, {session}')
    axs.set_xlabel('time relative to stimulation (s)')
    axs.set_ylabel('cross-validated distance (a.u.)')
    axs.axvline(0, color='k', lw=.8, ls='--')

    axs.axvline(0, color='k', ls='-', lw=.8)
    axs.axvline(.2, color='k', ls=':', lw=.8)
    axs.axvline(.5, color='k', ls='--', lw=.8)
    axs.axvline(1, color='k', ls='-.', lw=.8)
    axs.axhline(0, color='k', ls='-', lw=.8)

    axs.set_yscale('symlog', linthresh=.1)

    axs.legend(loc='upper left')

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', f'dist.timec.force.{participant_id}.{session}.png'))

    if make_plot:
        plt.show()
