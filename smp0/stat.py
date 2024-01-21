import numpy as np
from PcmPy.matrix import indicator
from .workflow import av_within_participant
import pandas as pd
from statsmodels.stats.anova import AnovaRM

def rm_anova(df, group_factors, anova_factors):
    """
    Perform repeated measures ANOVA for specified group and ANOVA factors.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    group_factors (list of str): Columns to group by.
    anova_factors (list of str): Factors to use in the ANOVA.

    Returns:
    pd.DataFrame: A DataFrame with the ANOVA results.
    """
    # Create an empty DataFrame to store results
    anova_results = pd.DataFrame(columns=['group', 'factor','F-value', 'pval', 'df', 'df_resid'])

    # Iterate over each combination of group factors
    for group_vals, df_group in df.groupby(group_factors):
        # Perform Repeated Measures ANOVA
        aovrm = AnovaRM(df_group, depvar='Value', subject='participant_id', within=anova_factors)
        res = aovrm.fit()

        # Extract results for each factor and interaction
        for factor in anova_factors + [':'.join(anova_factors)]:
            if factor in res.anova_table.index:
                F_value = res.anova_table.loc[factor, 'F Value']
                p_value = res.anova_table.loc[factor, 'Pr > F']
                df1 = res.anova_table.loc[factor, 'Num DF']
                df2 = res.anova_table.loc[factor, 'Den DF']

                # Append results to the DataFrame
                row = {'group': group_vals, 'factor': factor, 'F-value': F_value, 'pval': p_value, 'df': df1, 'df_resid': df2}
                anova_results.loc[len(anova_results)] = row

    return anova_results


class Anova3D:

    def __init__(self, data, channels, conditions, labels):

        self.data = data  # list of instances of class Dataset3D
        self.channels = channels
        self.conditions = conditions
        self.labels = labels

    def make_df(self, labels):
        n_conditions = len(self.conditions)
        n_timepoints = self.data[0].timepoints
        df = pd.DataFrame(columns=['participant_id', 'condition', 'channel', 'timepoint', 'Value'])
        participants = [self.data[p].descriptors['participant_id'] for p in range(len(self.data))]
        for p, p_data in enumerate(self.data):
            Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
            M, cond_names = av_within_participant(p_data.measurements, Z, cond_name=labels)
            for ch, channel in enumerate(p_data.channel_descriptors['channels']):
                for c, cond in enumerate(cond_names):
                    for tp in range(n_timepoints):
                        df.loc[len(df)] = {
                            'participant_id': p_data.descriptors['participant_id'],
                            'condition': cond,
                            'channel': channel,
                            'timepoint': tp,
                            'Value': M[c, ch, tp]
                        }
        return df


    # def rm_anova(self, data, labels):
    #     n_conditions = len(self.conditions)
    #     n_timepoints = data.shape[-1]
    #
    #     # fval = np.zeros((n_conditions, n_timepoints))
    #     # num_df = np.zeros((n_conditions, n_timepoints))
    #     # den_df = np.zeros((n_conditions, n_timepoints))
    #     # pval = np.zeros((n_conditions, n_timepoints))
    #
    #     for c, cond in enumerate(self.conditions):
    #         for tp in range(n_timepoints):
    #             n_participants = data.shape[0]
    #             n_labels = len(labels[c])
    #             subject_ids = np.repeat(np.arange(n_participants), n_labels)
    #             data_df = data[:, c, :, tp].flatten()
    #             labels_df = np.array([labels[c]] * n_participants).flatten()
    #
    #             df = pd.DataFrame({'Subject': subject_ids, 'Condition': labels_df, 'Score': data_df})
    #
    #             aovrm = AnovaRM(df, 'Score', 'Subject', within=['Condition'])
    #             res = aovrm.fit()
    #
    #             fval[c, tp] = res.anova_table['F Value'].to_numpy()[0]
    #             num_df[c, tp] = res.anova_table['Num DF'].to_numpy()[0]
    #             den_df[c, tp] = res.anova_table['Den DF'].to_numpy()[0]
    #             pval[c, tp] = res.anova_table['Pr > F'].to_numpy()[0]
    #
    #     return fval, num_df, den_df, pval
                # print(res.summary())

    # def _make_df_for_anova(self, data, condition):

    # def av_across_participants(self):
    #
    #     n_conditions = len(self.conditions)
    #
    #     channels_dict = {ch: [] for ch in self.channels}
    #     N = len(self.data)
    #     for p_data in self.data:
    #         Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
    #         M = av_within_participant(p_data.measurements, Z)
    #
    #         for ch in self.channels:
    #             if ch in p_data.channel_descriptors['channels']:
    #                 channel_index = p_data.channel_descriptors['channels'].index(ch)
    #                 channels_dict[ch].append(M[:, channel_index])
    #
    #     Mean, SD, SE = {}, {}, {}
    #     for ch in self.channels:
    #         channel_data = np.array(channels_dict[ch])
    #         channels_dict[ch] = channel_data.reshape((channel_data.shape[0], n_conditions, int(channel_data.shape[1] / n_conditions), channel_data.shape[2]))
    #         Mean[ch] = np.mean(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
    #                                                           channel_data.shape[2]))
    #         SD[ch] = np.std(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
    #                                                        channel_data.shape[2]))
    #         SE[ch] = (SD[ch] / np.sqrt(N)).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
    #                                                 channel_data.shape[2]))
    #
    #     return Mean, SD, SE, channels_dict
