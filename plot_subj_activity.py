import argparse
import os
import json

import nibabel as nb

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_1samp
import numpy as np
import globals as gl
import rsatoolbox as rsa

import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--participant_id', default='subj101', help='Participant ID')
    parser.add_argument('--atlas', default='BA_exvivo', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere')
    parser.add_argument('--glm', default='1', help='GLM model')
    parser.add_argument('--epoch', default='plan', help='Selected epoch')
    parser.add_argument('--instr', default='nogo', help='Selected instruction')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    Hem = args.Hem
    glm = args.glm
    sel_epoch = args.epoch
    sel_instr = args.instr

    experiment = 'smp1'

    path = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id, 'cont')
    pathSurf = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id)

    rois = {'BA_exvivo': ['BA1_exvivo',
                          'BA2_exvivo',
                          'BA3a_exvivo',
                          'BA3b_exvivo',
                          'BA4a_exvivo',
                          'BA4p_exvivo',
                          'BA6_exvivo'],
            'aparc': ['precentral',
                      'postcentral',
                      'caudalmiddlefrontal',
                      'pericalcarine']}
    rois = rois[atlas]

    A = nb.load(os.path.join(pathSurf, f'{participant_id}.{Hem}.32k.{atlas}.label.gii'))
    keys = pd.Series(A.darrays[0].data)
    label = [(l.key, l.label) for l in A.labeltable.labels]
    label_df = pd.DataFrame(label, columns=['key', 'label'])
    mapping_dict = dict(zip(label_df['key'], label_df['label']))
    labels = keys.replace(mapping_dict)

    B = nb.load(os.path.join(path, f'cont.{Hem}.func.gii'))
    col = [con.metadata['Name'] for con in B.darrays]
    idx_exec = col.index('con_exec-.nii')
    idx_plan_nogo = col.index('con_plan_nogo-.nii')

    exec = B.darrays[idx_exec].data
    plan_nogo = B.darrays[idx_plan_nogo].data

    plan_nogo_df = pd.DataFrame({
        'activity': plan_nogo,
        'epoch': 'plan_nogo',
        'labels': labels})
    exec_df = pd.DataFrame({
        'activity': exec,
        'epoch': 'exec',
        'labels': labels}
    )

    df = pd.concat([plan_nogo_df, exec_df])
    df = df[df['labels'].isin(rois)]

    fig, axs = plt.subplots(figsize=(4, 5))

    sns.barplot(data=df, ax=axs, palette=['blue', 'red'], x='labels', y='activity', hue='epoch')

    # axs.bar(rois, mdist)
    axs.set_ylabel('activity (a.u.)')
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right')  # Correct rotation method
    axs.set_ylim([-2.5, 2.5])

    # # Adding asterisks for significant results
    # significance_level = 0.05
    # for idx, p in enumerate(p_values):
    #     if p < significance_level:
    #         axs.text(idx, mdist[idx], '*', ha='center', va='bottom', color='k', fontsize=12)

    fig.subplots_adjust(left=.20, bottom=.25)

    axs.set_title(f'{participant_id}\nepoch:{sel_epoch}, instr:{sel_instr}')

    plt.show()