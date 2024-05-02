import argparse
import os
import nibabel as nb
import matplotlib.pyplot as plt

import sys

sys.path.append('/Users/mnlmrc/Documents/GitHub')

import globals as gl

import surfAnalysisPy as surf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--atlas', default='ROI', help='Atlas name')

    args = parser.parse_args()

    atlas = args.atlas
    borders = ['/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.L.border',
               '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.R.border']

    experiment = 'smp1'

    pathAtlas = os.path.join(gl.baseDir, experiment, 'atlases')

    Hem = ['L', 'R']
    hemispheres = ['Left hemisphere', 'Right hemisphere']
    surface = ['fs32k_L', 'fs32k_R']

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for h in range(2):
        A = nb.load(os.path.join(pathAtlas, f'{atlas}.32k.{Hem[h]}.label.gii'))
        plt.sca(axs[h])
        surf.plot.plotmap(A.darrays[0].data, surface[h],
                          underlay=None,
                          cmap='tab20',
                          borders=borders[h],
                          underscale=[-1.5, 1],
                          alpha=.8,
                          new_figure=False,
                          colorbar=False)
        plt.title(hemispheres[h])

    plt.suptitle(f'atlas:{atlas}')
    plt.savefig(os.path.join(gl.baseDir, experiment, 'atlases',
                             f"flat_{atlas}" + '.png'), dpi=600)
    plt.show()


