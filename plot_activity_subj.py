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
    parser.add_argument('--participant_id', default='subj100', help='Participant ID')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--Hem', default='L', help='Hemisphere')
    parser.add_argument('--glm', default='4', help='GLM model')

    args = parser.parse_args()

    participant_id = args.participant_id
    atlas = args.atlas
    Hem = args.Hem
    glm = args.glm

    experiment = 'smp1'

    path = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id)
    pathSurf = os.path.join(gl.baseDir, experiment, gl.wbDir, participant_id)
    pathAtlas = os.path.join(gl.baseDir, experiment, 'atlases')

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
                      'pericalcarine'],
            'ROI': ['PMd',
                    'PMv',
                    'M1',
                    'S1',
                    'SPLa',
                    'SPLp',
                    'V1']}

    rois = rois[atlas]

    A = nb.load(os.path.join(pathAtlas, f'{atlas}.32k.{Hem}.label.gii'))
    keys = pd.Series(A.darrays[0].data)
    label = [(l.key, l.label) for l in A.labeltable.labels[1:]]
    label_df = pd.DataFrame(label, columns=['key', 'label'])
    mapping_dict = dict(zip(label_df['key'], label_df['label']))
    labels = keys.replace(mapping_dict)

    B = nb.load(os.path.join(path, f'glm{glm}.con.{Hem}.func.gii'))
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
    df = df[df['labels'].isin(rois) & (df['activity'].abs() > .1)]

    fig, axs = plt.subplots(figsize=(6, 5))

    sns.barplot(data=df, ax=axs, palette=['blue', 'red'], x='labels', y='activity', hue='epoch',
                   order=rois)

    # axs.bar(rois, mdist)
    axs.set_ylabel('% signal change')
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right')  # Correct rotation method
    axs.set_xlim((axs.get_xlim()[0], axs.get_xlim()[1]))
    axs.axhline(0, axs.get_xlim()[0], axs.get_xlim()[1], color='k', lw=.8)
    axs.set_ylim([-40, 40])
    axs.legend(loc='lower left')
    # axs.set_ylim([-1, 1])

    # # Adding asterisks for significant results
    # significance_level = 0.05
    # for idx, p in enumerate(p_values):
    #     if p < significance_level:
    #         axs.text(idx, mdist[idx], '*', ha='center', va='bottom', color='k', fontsize=12)

    fig.subplots_adjust(left=.20, bottom=.25)

    axs.set_title(f'{participant_id}\nactivity')

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id, f'activity.{atlas}.png'))

    plt.show()
