import pandas as pd


def vlookup_value(df, search_column, search_value, target_column):
    """
    Returns the value from 'target_column' where 'search_column' equals 'search_value'.

    :param df: Pandas DataFrame to search in.
    :param search_column: Column name in which to search for 'search_value'.
    :param search_value: Value to search for in the 'search_column'.
    :param target_column: Column from which to return the value.
    :return: Value from 'target_column' or None if 'search_value' is not found.
    """
    matching_rows = df[df[search_column] == search_value]
    if not matching_rows.empty:
        return matching_rows.iloc[0][target_column]
    else:
        return None
