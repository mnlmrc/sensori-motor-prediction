import numpy as np
from PcmPy.matrix import indicator
from .workflow import av_within_participant


class Anova:

    def __init__(self, data, channels, conditions, labels):

        self.data = data
        self.channels = channels
        self.conditions = conditions
        self.labels = labels

    def rm_anova_over_timepoints(self, data, labels):

        for c, cond in enumerate(self.conditions):
            for ch, channel in enumerate(self.channels):
                for tp in range(data.shape[-1]):
                    n_participants = data[channel].shape[0]
                    n_labels = len(labels)

                    subject_ids = np.repeat(np.arange(n_participants), n_labels)
                    data_df = data[channel][:, c, :, tp].flatten()
                    levels_df = np.array([levels] * n_participants).flatten()

                    df = pd.DataFrame({'Subject': subject_ids, 'Condition': levels_df, 'Score': data_df})

                    aovrm = AnovaRM(df, 'Score', 'Subject', within=['Condition'])
                    res = aovrm.fit()

                    print(res.summary())

    # def _make_df_for_anova(self, data, condition):

    def _av_across_participants(self):

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
            channels_dict[ch] = channel_data.reshape((n_participants, n_conditions,
                                                      int(channel_data.shape[1] / n_conditions), channel_data.shape[2]))
            Mean[ch] = np.mean(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                              channel_data.shape[2]))
            SD[ch] = np.std(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                           channel_data.shape[2]))
            SE[ch] = (SD[ch] / np.sqrt(N)).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                    channel_data.shape[2]))

        return Mean, SD, SE, channels_dict
