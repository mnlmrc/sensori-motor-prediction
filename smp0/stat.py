import numpy as np
from PcmPy.matrix import indicator
from .workflow import av_within_participant
import pandas as pd
import itertools
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel


def pairwise(df, factors, dep_var='Value', alpha=0.05):
    """
    Perform post hoc tests for significant factors and interactions in repeated measures ANOVA.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    dependent_var (str): The dependent variable.
    factors (list of tuples): Each tuple contains the names of the factors for the interaction.
    alpha (float): Significance level for the tests.

    Returns:
    pd.DataFrame: A DataFrame with the post hoc test results.
    """
    res = pd.DataFrame(columns=['comparison', 'stat', 'p-adj'])

    for factor_group in factors:
        # Create interaction term if necessary
        if len(factor_group) > 1:
            df['interaction'] = df[list(factor_group)].astype(str).agg('-'.join, axis=1)
            factor_to_test = 'interaction'
        else:
            factor_to_test = factor_group[0]

        # Generate all pairwise combinations for the factor levels
        levels = df[factor_to_test].unique()
        for level1, level2 in itertools.combinations(levels, 2):
            group1 = df[df[factor_to_test] == level1][dep_var]
            group2 = df[df[factor_to_test] == level2][dep_var]

            # Perform the paired t-test
            stat, p = ttest_rel(group1, group2)

            # Bonferroni correction
            p_adj = p * (len(levels) * (len(levels) - 1) / 2)  # Adjust for the number of comparisons

            # Append results to the DataFrame
            comparison = f"{factor_to_test}: {level1} vs {level2}"
            res.loc[len(res)] = {'comparison': comparison, 'stat': stat, 'p-adj': p_adj}

        # Drop interaction term to clean up for next iteration
        if 'interaction' in df.columns:
            df.drop('interaction', axis=1, inplace=True)

    return res


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
    anova_results = pd.DataFrame(columns=['group', 'factor', 'F-value', 'pval', 'df', 'df_resid'])

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
                row = {'group': group_vals, 'factor': factor, 'F-value': F_value, 'pval': p_value, 'df': df1,
                       'df_resid': df2}
                anova_results.loc[len(anova_results)] = row

    return anova_results


class Stats:

    def __init__(self, data, channels, conditions, labels):

        self.data = data  # list of instances of class Dataset3D
        self.channels = channels
        self.conditions = conditions
        self.labels = labels
        self.df = self.make_df()

    def make_df(self):
        n_conditions = len(self.conditions)
        n_timepoints = self.data[0].timepoints
        df = pd.DataFrame(columns=['participant_id', 'condition', 'channel', 'timepoint', 'Value'])
        participants = [self.data[p].descriptors['participant_id'] for p in range(len(self.data))]
        for p, p_data in enumerate(self.data):
            Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
            M, cond_names = av_within_participant(p_data.measurements, Z, cond_name=self.labels)
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

    def rm_anova(self, group_factors, anova_factors):
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
        anova_results = pd.DataFrame(columns=['group', 'factor', 'F-value', 'pval', 'df', 'df_resid'])

        # Iterate over each combination of group factors
        for group_vals, df_group in self.df.groupby(group_factors):
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
                    row = {'group': group_vals, 'factor': factor, 'F-value': F_value, 'pval': p_value, 'df': df1,
                           'df_resid': df2}
                    anova_results.loc[len(anova_results)] = row

        return anova_results

    def pairwise(self, group_factors, test_factors):

        group = list()
        for g in group_factors:
            group.append(self.df[g].unique())

        for comb in itertools.product(*group):
            df_in = self.df
            for g in group_factors:
                df_in = df_in[df_in[g] == comb[g]]

    def _pairwise(self, factors, dep_var='Value', alpha=0.05):

        """
        Perform post hoc tests for significant factors and interactions in repeated measures ANOVA.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        dependent_var (str): The dependent variable.
        factors (list of tuples): Each tuple contains the names of the factors for the interaction.
        alpha (float): Significance level for the tests.

        Returns:
        pd.DataFrame: A DataFrame with the post hoc test results.
        """
        res = pd.DataFrame(columns=['comparison', 'stat', 'p-adj'])

        for factor_group in factors:
            # Create interaction term if necessary
            if len(factor_group) > 1:
                self.df['interaction'] = self.df[list(factor_group)].astype(str).agg('-'.join, axis=1)
                factor_to_test = 'interaction'
            else:
                factor_to_test = factor_group[0]

            # Generate all pairwise combinations for the factor levels
            levels = self.df[factor_to_test].unique()
            for level1, level2 in itertools.combinations(levels, 2):
                group1 = self.df[self.df[factor_to_test] == level1][dep_var]
                group2 = self.df[self.df[factor_to_test] == level2][dep_var]

                # Perform the paired t-test
                stat, p = ttest_rel(group1, group2)

                # Bonferroni correction
                p_adj = p * (len(levels) * (len(levels) - 1) / 2)  # Adjust for the number of comparisons

                # Append results to the DataFrame
                comparison = f"{factor_to_test}: {level1} vs {level2}"
                res.loc[len(res)] = {'comparison': comparison, 'stat': stat, 'p-adj': p_adj}

            # Drop interaction term to clean up for next iteration
            if 'interaction' in self.df.columns:
                self.df.drop('interaction', axis=1, inplace=True)

        return res
