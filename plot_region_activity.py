import os
import numpy as np
import nibabel as nb
import matplotlib

matplotlib.use('macOSX')
import matplotlib.pyplot as plt
import globals as gl

# Initialize subject list and type
subj_id = 'subj100'
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

activity = {
    'label': [[k.label for k in key[0]], [k.label for k in key[1]]],
    'key': [[k.key for k in key[0]], [k.key for k in key[1]]],
    'condition': [cond.metadata['Name'] for cond in D[0].darrays],
    'value': np.zeros((ncols, len(Hem), len(key[0])))
}

# Process each hemisphere
for h, hem in enumerate(Hem):
    LD = ldata[h]

    for cond, Dd in enumerate(D[h].darrays):
        data = Dd.data
        col = Dd.meta.metadata['Name']

        for reg, k in enumerate(activity['key'][h]):
            print(f'Averaging data - column: {col}, hemisphere: {hem}, area: {activity["label"][h][reg]}')
            activity['value'][cond, h, reg] = data[LD == k].mean()

fig, axs = plt.subplots(2, sharey=True, sharex=True)
title = ['Left hemisphere', 'Right hemisphere']
for h in range(len(Hem)):
    x = activity['label'][h]
    y = activity['condition']
    z = activity['value'][:, h]
    # axs[h].contourf(x, y, z, vmin=np.nanmin(activity['value']), vmax=np.nanmax(activity['value']))
    axs[h].imshow(z)
    axs[h].set_title(title[h])


axs[-1].set_xticks(np.linspace(0, reg, reg+1))
axs[-1].set_xticklabels(activity['label'][-1],
                        rotation=90, ha='center', va='top')
axs[-1].set_yticks(np.linspace(0, cond, cond+1))
axs[-1].set_yticklabels(activity['condition'])


