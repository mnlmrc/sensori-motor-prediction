import sys

sys.path.append('/Users/mnlmrc/Documents/GitHub')
import numpy as np
import nibabel as nb
# import nitools as nt
import os
import matplotlib.pyplot as plt
import globals as gl

import surfAnalysisPy as surf

if __name__ == "__main__":
    subj_id = sys.argv[1]
    typ = sys.argv[2]
    experiment = 'smp1'

    dataL = f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/surfaceWB/{subj_id}/{typ}/{typ}.L.func.gii'
    dataR = f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/surfaceWB/{subj_id}/{typ}/{typ}.R.func.gii'
    bordersL = '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.L.border'
    underlayL = '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.LR.sulc.dscalar.nii'
    bordersR = '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.R.border'
    underlayR = '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.LR.sulc.dscalar.nii'

    cscale = {
        'tval': [-2.5, 2.5],
        'psc': [-2, 2]
    }

    DL = nb.load(dataL)
    DR = nb.load(dataR)

    for im in range(DL.numDA):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plt.sca(axs[0])
        surf.plot.plotmap(DL.darrays[im].data, 'fs32k_L',
                          underlay=None,
                          borders=bordersL,
                          cscale=cscale[typ],
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False)
        plt.title('Left hemisphere')
        plt.sca(axs[1])
        surf.plot.plotmap(DR.darrays[im].data, 'fs32k_R',
                          underlay=None,
                          borders=bordersR,
                          cscale=cscale[typ],
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False)
        plt.suptitle(f"{subj_id} - {DL.darrays[im].metadata['Name']}")
        plt.title('Right hemisphere')

        plt.subplots_adjust(bottom=0, left=0, wspace=0, hspace=0, right=1)

        plt.savefig(os.path.join(gl.baseDir, experiment, 'figures', subj_id, f"{DL.darrays[im].metadata['Name']}".rstrip('.nii') + '.png'))



