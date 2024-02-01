import sys

import pandas as pd

from smp0.globals import base_dir
from smp0.sinergies import nnmf
from smp0.visual import make_colors
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    datatype = sys.argv[1]

    participants = [100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110]

    file_path = base_dir + f"/smp0/smp0_{datatype}_binned.stat"
    data = pd.read_csv(file_path)
    data = data[data['participant_id'].isin(participants)]

    channels = data['channel'].unique()
    timepoints = data['timepoint'].unique()
    stimFingers = data['stimFinger'].unique()
    cues = ['0%', '25%', '50%', '75%', '100%']

    colors = make_colors(2, ecol=('orange', 'purple'))
    labels = ['coefficient #1 (index-like)', 'coefficient #2 (ring-like)']
    palette = {labels: color for labels, color in zip(labels, colors)}

    R_squared = pd.DataFrame(columns=['participant_id', 'timepoint', 'Rsquared'])
    for p, participant_id in enumerate(participants):
        for tp in timepoints[1:]:
            pdata = data[(data['participant_id'] == participant_id) & (data['timepoint'] == tp)]
            pchannels = pdata['channel'].unique().tolist()
            n_pchannels = len(pchannels)

            if len(pdata) % n_pchannels == 0:
                X = pdata['Value'].to_numpy().reshape((-1, n_pchannels))
                _, _, R = nnmf(X)

                R_squared.loc[len(R_squared)] = {
                        'participant_id': str(participant_id),
                        'timepoint': tp,
                        'Rsquared': R
                }

    fig, axs = plt.subplots()
    sns.barplot(ax=axs, data=R_squared, x='timepoint', y='Rsquared', errorbar='se',
                color='red', legend=None, hue_order=labels, err_kws={'color': 'k'})


