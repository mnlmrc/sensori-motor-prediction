import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp

from smp1.fetch import load_dat
from smp1.globals import base_dir

if __name__ == "__main__":

    experiment = "smp1"
    participant_id = "100"

    D = load_dat(experiment, participant_id)

    forceDiff_avgs = D.groupby('BN')['forceDiff'].mean()
    forceDiff_avg = D['forceDiff'].mean()
    forceDiff_std = D['forceDiff'].std()

    t_test = ttest_1samp(D['forceDiff'], 0)

    forceDiff_avgs_norm = (forceDiff_avgs - forceDiff_avgs.min()) / (
                forceDiff_avgs.max() - forceDiff_avgs.min())

    cmap = plt.get_cmap('coolwarm')
    colors = cmap(forceDiff_avgs_norm)

    # Create a custom palette
    palette = {str(bn): color for bn, color in zip(forceDiff_avgs.index, colors)}

    # Plotting using seaborn with the custom palette
    fig, axs = plt.subplots()
    sns.boxplot(x='BN', y='forceDiff', data=D, palette=palette)
    axs.set_title(f'Cue driven preactivation, participant {participant_id}, training session')
    axs.set_xlabel('Block number')
    axs.set_ylabel('forceDiff (cued vs. uncued) [N]')

    # Adding text for overall average, standard deviation, and t-test result
    textstr = '\n'.join((
        f'Overall Average: {forceDiff_avg:.4f}',
        f'Standard Deviation: {forceDiff_std:.4f}',
        f'T-test result: t={t_test.statistic:.4f}, p={t_test.pvalue:.4f}'))

    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place a text box in upper left in axes coords
    plt.text(0.5, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.show()

    fig.savefig(f'{base_dir}/smp1/figures/smp1_training_{participant_id}.png')
