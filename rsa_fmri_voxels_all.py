import argparse
import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
import rsatoolbox as rsa

import mat73
from scipy.io import loadmat

from scipy.linalg import sqrtm

from utils import sort_key

import globals as gl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--participants', nargs='+', default=[
        'subj100',
        'subj101',
        'subj102',
        'subj103',
        'subj105',
        'subj106'
    ],
                        help='Participant ID (e.g., subj100, subj101, ...)')
    parser.add_argument('--atlas', default='ROI', help='Atlas name')
    parser.add_argument('--glm', default='8', help='GLM model (e.g., 1, 2, ...)')
    # order:
    # glm8 : [2, 4, 6, 1,  0, 3, 5, 7]
    # glm9 : [0, 4, 7, 10, 2,  5, 8, 11, 3,  1, 6, 9, 12]
    parser.add_argument('--index', nargs='+', type=int, default=[2, 4, 6, 1,  0, 3, 5, 7],
                        help='Label order')

    args = parser.parse_args()

    participants = args.participants
    atlas = args.atlas
    glm = args.glm
    index = args.index

    os.system('pwd')

    for participant in participants:
        os.system(f'python3 rsa_fmri_voxels_subj.py --participant_id {participant} --glm {glm}')


