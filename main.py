import os

import matplotlib.pyplot as plt
import pandas as pd

from force import calculate_difference
import globals as gl
from util import pad_dict_values
import seaborn as sns


def main(action):
    match action:
        case "behavioural:75-25":
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

            index_padded = pad_dict_values(index_diff)
            ring_padded = pad_dict_values(ring_diff)

            df_index = pd.DataFrame(index_padded).reset_index().melt(id_vars='index', var_name='Session',
                                                                     value_name='forceDiff').drop('index', axis=1)
            df_index['stimFinger'] = 'index'
            df_ring = pd.DataFrame(ring_padded).reset_index().melt(id_vars='index', var_name='Session',
                                                                   value_name='forceDiff').drop('index', axis=1)
            df_ring['stimFinger'] = 'ring'

            return df_index, df_ring

        case "plot:75-25":

            df_index, df_ring = main("behavioural:75-25")

            df_combined = pd.concat([df_index, df_ring]).reset_index()

            fig, axs = plt.subplots(figsize=(3.5, 6))
            sns.boxplot(data=df_combined, x='Session', y='forceDiff', hue='stimFinger',  palette=['green', 'red'])
            axs.axhline(0, lw='.8', color='k')
            axs.set_yscale('symlog')
            axs.set_ylabel('force difference (N)')
            axs.set_xlabel('')

            # Calculate medians by session and finger
            medians = df_combined.groupby(['Session', 'stimFinger'])['forceDiff'].median().reset_index()

            # Add median annotations above the median line
            for median in medians.itertuples():

                print(f"session:{median.Session}, stimFinger:{median.stimFinger}, forceDiff={median.forceDiff:.2f}")

            # Show the plot
            axs.set_title('Force difference between 75% and 25%\nacross sessions (0.2-0.4s after perturbation)')

            fig.subplots_adjust(left=.3)
            plt.show()


sessions = ['behav', 'training', 'scanning']
participants = {
    'behav': ['subj100', 'subj101', 'subj102', 'subj103', 'subj104', 'subj105', 'subj106', 'subj107', 'subj108',
              'subj109', 'subj110'],
    'training': ['subj100', 'subj101', 'subj102', 'subj103', 'subj104', 'subj105', 'subj106'],
    'scanning': ['subj100', 'subj101', 'subj102', 'subj103', 'subj104', 'subj105', 'subj106']
}

# main('plot:75-25')
