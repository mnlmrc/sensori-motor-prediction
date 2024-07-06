import os

import pandas as pd
from force import calculate_difference
import globals as gl

session = ['behav', 'training', 'scanning']
participant = {
    'behav': ['subj100',
              'subj101',
              'subj102',
              'subj103',
              'subj104',
              'subj105',
              'subj106',
              'subj107',
              'subj108',
              'subj109',
              'subj110'],
    'training': ['subj100',
                 'subj101',
                 'subj102',
                 'subj103',
                 'subj104',
                 'subj105',
                 'subj106', ],
    'scanning': ['subj100',
                 'subj101',
                 'subj102',
                 'subj103',
                 'subj104',
                 'subj105',
                 'subj106', ]
}

index_diff = {
    'behav': [],
    'training': [],
    'scanning': []
}

ring_diff = {
    'behav': [],
    'training': [],
    'scanning': []
}

for sess in session:
    participant_id = participant[sess]

    for p in participant_id:
        sn = int(''.join([c for c in p if c.isdigit()]))
        if sess is 'behav':
            experiment = 'smp0'
            path = os.path.join(gl.baseDir, experiment, p, 'mov')
            data = pd.read_csv(os.path.join(path, f'{experiment}_{sn}_binned.tsv'), sep='\t')
        elif sess is 'scanning':
            experiment = 'smp1'
            path = os.path.join(gl.baseDir, experiment, gl.behavDir, p)
            data = pd.read_csv(os.path.join(path, f'{experiment}_{sn}_binned.tsv'), sep='\t')

        # Define parameters
        timewin = 'LLR'
        column = 'ring'

        # Calculate differences
        index_diff[sess].append(calculate_difference(data, timewin, 'index', column))
        ring_diff[sess].append(calculate_difference(data, timewin, 'ring', column))
