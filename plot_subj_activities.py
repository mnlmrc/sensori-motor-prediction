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
    parser.add_argument('--atlas', default='aparc', help='Atlas name')
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

    A = nb.load(os.path.join(path, f'cont.{Hem}.func.gii'))
    col = [con.metadata['Name'] for con in A.darrays]
    idx_exec = col.index('con_exec-.nii')
    idx_plan = col.index('con_plan_nogo-.nii')
