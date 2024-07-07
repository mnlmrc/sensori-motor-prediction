import os
import numpy as np
import pandas as pd
from force import calculate_difference
import globals as gl
import matplotlib.pyplot as plt
import seaborn as sns





# Define session and participant data
sessions = ['behav', 'training', 'scanning']
participants = {
    'behav': ['subj100', 'subj101', 'subj102', 'subj103', 'subj104', 'subj105', 'subj106', 'subj107', 'subj108',
              'subj109', 'subj110'],
    'training': ['subj100', 'subj101', 'subj102', 'subj103', 'subj104', 'subj105', 'subj106'],
    'scanning': ['subj100', 'subj101', 'subj102', 'subj103', 'subj104', 'subj105', 'subj106']
}

index_diff = {session: [] for session in sessions}
ring_diff = {session: [] for session in sessions}

# Process each session and participant
for session in sessions:
    for participant in participants[session]:
        sn = int(''.join(filter(str.isdigit, participant)))
        if session == 'behav':
            experiment, path = 'smp0', os.path.join(gl.baseDir, 'smp0', participant, 'mov')
        else:
            experiment = 'smp1'
            dir_path = gl.behavDir if session == 'scanning' else gl.trainDir
            path = os.path.join(gl.baseDir, 'smp1', dir_path, participant)

        data = pd.read_csv(os.path.join(path, f'{experiment}_{sn}_binned.tsv'), sep='\t')

        # Calculate differences
        index_diff[session].append(calculate_difference(data, 'LLR', 'index', 'index'))
        ring_diff[session].append(calculate_difference(data, 'LLR', 'ring', 'ring'))

# Pad the differences
index_padded = pad_dict_values(index_diff)
ring_padded = pad_dict_values(ring_diff)

# Convert dictionaries to DataFrame and melt them
df_index = pd.DataFrame(index_padded).reset_index().melt(id_vars='index', var_name='Session',
                                                         value_name='forceDiff').drop('index', axis=1)
df_index['stimFinger'] = 'Index'

df_ring = pd.DataFrame(ring_padded).reset_index().melt(id_vars='index', var_name='Session',
                                                       value_name='forceDiff').drop('index', axis=1)
df_ring['stimFinger'] = 'Ring'

# Combine the melted DataFrames
df_combined = pd.concat([df_index, df_ring])

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Session', y='forceDiff', hue='stimFinger', data=df_combined, palette=['green'])
sns.boxplot(x='Session', y='forceDiff', hue='stimFinger', data=df_combined, palette=['red'])

# Show the plot
plt.title('Force difference between 75% and 25% across sessions')
plt.show()
