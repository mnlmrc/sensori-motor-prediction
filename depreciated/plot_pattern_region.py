import os
import numpy as np
import nibabel as nb
import matplotlib

matplotlib.use('macOSX')
import matplotlib.pyplot as plt
import globals as gl

# Initialize subject list and type
subj_id = 'subj101'
experiment = 'smp1'
type = 'psc'

actDir = os.path.join(gl.baseDir, experiment, gl.wbDir, subj_id, type)

print('Loading label files...')
label = [
    nb.load(os.path.join(gl.baseDir, experiment, gl.wbDir, subj_id, f'{subj_id}.L.32k.label.gii')),
    nb.load(os.path.join(gl.baseDir, experiment, gl.wbDir, subj_id, f'{subj_id}.R.32k.label.gii'))
]

key = [
    label[0].labeltable.labels,
    label[1].labeltable.labels
]

ldata = [
    label[0].darrays[0].data,
    label[1].darrays[0].data
]

Hem = ['L', 'R']

print(f'Loading activity data...')
D = [
    nb.load(os.path.join(actDir, f'{type}.L.func.gii')),
    nb.load(os.path.join(actDir, f'{type}.R.func.gii'))
]

ncols = D[0].numDA

condition = [
    'psc_plan_nogo_cue000-.nii',
    'psc_plan_nogo_cue025-.nii',
    'psc_plan_nogo_cue050-.nii',
    'psc_plan_nogo_cue075-.nii',
    'psc_plan_nogo_cue100-.nii'
]

rois = [
    'precentral',
    'postcentral',
    'superiorfrontal',
    'rostralmiddlefrontal',
    'caudalmiddlefrontal',
    'parsopercularis'
]

roisKeyAll = ([k.label for k in key[0]], [k.label for k in key[1]])
roisFilt = ([roisKeyAll[0].index(r) for r in roisKeyAll[0] if r in rois],
            [roisKeyAll[1].index(r) for r in roisKeyAll[1] if r in rois])

roisNameAll = [[k.label for k in key[0]], [k.label for k in key[1]]]
roisNameFilt = ([roisNameAll[0].index(r) for r in roisNameAll[0] if r in rois],
                [roisNameAll[1].index(r) for r in roisNameAll[1] if r in rois])

condAll = [cond.metadata['Name'] for cond in D[0].darrays]
condFilt = [c for c in condAll if c in condition]

activity = {
    'label': roisNameFilt,
    'key': roisFilt,
    'condition': condFilt,
    'value': np.zeros((len(condFilt), len(Hem), len(rois)))
}

# Process each hemisphere
for h, hem in enumerate(Hem):
    LD = ldata[h]
    cond = 0
    for Dd in D[h].darrays:
        data = Dd.data
        col = Dd.meta.metadata['Name']
        if col in activity['condition']:
            for reg, k in enumerate(activity['key'][h]):
                print(f'Averaging data - column: {col}, hemisphere: {hem}, area: {activity["label"][h][reg]}')
                activity['value'][cond, h, reg] = data[LD == k].mean()
            cond = cond + 1

fig, axs = plt.subplots(2, sharey=True, sharex=True)
title = ['Left hemisphere', 'Right hemisphere']
for h in range(len(Hem)):
    axs[h].plot(activity['condition'], activity['value'][:, h], marker='o')
    axs[h].set_title(title[h])
    axs[h].set_xticks(np.arange(len(activity['condition'])))  # Setting x-tick positions
    axs[h].set_xticklabels(activity['condition'], rotation=90)  # Rotate xticklabels

axs[0].legend(rois, ncols=np.floor(len(rois)/2))
fig.supylabel('PSC (a.u.)')

