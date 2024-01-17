import numpy as np
from PcmPy.matrix import indicator
from .workflow import av_within_participant
import pandas as pd
from statsmodels.stats.anova import AnovaRM


class Anova:

    def __init__(self, data, channels, conditions, labels):

        self.data = data
        self.channels = channels
        self.conditions = conditions
        self.labels = labels

    def rm_anova_over_timepoints(self, data, labels):
        n_conditions = len(self.conditions)
        n_timepoints = data.shape[-1]

        fval = np.zeros((n_conditions, n_timepoints))
        num_df = np.zeros((n_conditions, n_timepoints))
        den_df = np.zeros((n_conditions, n_timepoints))
        pval = np.zeros((n_conditions, n_timepoints))

        for c in range(n_conditions):
            for tp in range(n_timepoints):
                n_participants = data.shape[0]
                n_labels = len(labels[c])
                subject_ids = np.repeat(np.arange(n_participants), n_labels)
                data_df = data[:, c, :, tp].flatten()
                labels_df = np.array([labels[c]] * n_participants).flatten()

                df = pd.DataFrame({'Subject': subject_ids, 'Condition': labels_df, 'Score': data_df})

                aovrm = AnovaRM(df, 'Score', 'Subject', within=['Condition'])
                res = aovrm.fit()

                fval[c, tp] = res.anova_table['F Value'].to_numpy()[0]
                num_df[c, tp] = res.anova_table['Num DF'].to_numpy()[0]
                den_df[c, tp] = res.anova_table['Den DF'].to_numpy()[0]
                pval[c, tp] = res.anova_table['Pr > F'].to_numpy()[0]

        return fval, num_df, den_df, pval
                # print(res.summary())

    # def _make_df_for_anova(self, data, condition):

    def av_across_participants(self):

        n_conditions = len(self.conditions)

        channels_dict = {ch: [] for ch in self.channels}
        N = len(self.data)
        for p_data in self.data:
            Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
            M = av_within_participant(p_data.measurements, Z)

            for ch in self.channels:
                if ch in p_data.channel_descriptors['channels']:
                    channel_index = p_data.channel_descriptors['channels'].index(ch)
                    channels_dict[ch].append(M[:, channel_index])

        Mean, SD, SE = {}, {}, {}
        for ch in self.channels:
            channel_data = np.array(channels_dict[ch])
            channels_dict[ch] = channel_data.reshape((channel_data.shape[0], n_conditions, int(channel_data.shape[1] / n_conditions), channel_data.shape[2]))
            Mean[ch] = np.mean(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                              channel_data.shape[2]))
            SD[ch] = np.std(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                           channel_data.shape[2]))
            SE[ch] = (SD[ch] / np.sqrt(N)).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                    channel_data.shape[2]))

        return Mean, SD, SE, channels_dict
