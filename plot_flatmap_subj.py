import os
import sys

import nibabel as nb

import globals as gl

import matplotlib.pyplot as plt

sys.path.append('/Users/mnlmrc/Documents/GitHub')
import surfAnalysisPy as surf

surface = ['fs32k_L', 'fs32k_R']

if os.path.isdir('/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh'):
    meshDir = '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh'
elif os.path.isdir('/home/ROBARTS/memanue5/Documents/GitHub/surfAnalysis/standard_mesh'):
    meshDir = '/home/ROBARTS/memanue5/Documents/GitHub/surfAnalysis/standard_mesh'
else:
    meshDir = None


borders = [f'{meshDir}/fs_L/fs_LR.32k.L.border', f'{meshDir}/fs_L/fs_LR.32k.R.border']
underlay = [f'{meshDir}/fs_L/fs_LR.32k.LR.sulc.dscalar.nii', f'{meshDir}fs_R/fs_LR.32k.LR.sulc.dscalar.nii']

Hem = ['L', 'R']

# fig, axs = plt.subplots()

for h, H in enumerate(Hem):
    mean = nb.load(os.path.join(gl.baseDir, 'smp1', gl.wbDir, 'group', f'mean.{H}.func.gii'))

    for img in mean.darrays:
        # plt.sca(axs)
        surf.plot.plotmap(img.data, surface[h],
                          underlay=None,
                          borders=borders[h],
                          # cscale=cscale[meas],
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=True,
                          colorbar=False)


