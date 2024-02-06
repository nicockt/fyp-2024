import pandas as pd

def z_score_standardization(data_column):
    """
    Z-score standardization for a given pandas Series (column).

    Parameters:
    - data_column: pandas Series

    Returns:
    - Z-score standardized column: pandas Series
    """

    mean_value = data_column.mean()
    std = data_column.std()

    standardized_column = (data_column - mean_value) / std

    return standardized_column


# TODO
def annual_to_quarterly_percentage_change(annual_percentage_changes):
    """
    Convert annual percentage changes to quarterly percentage changes using the formula:
    (1 + Growth Rate)^(1/4) - 1

    Parameters:
    - annual_percentage_changes (%): pandas Series

    Returns:
    - Quarterly percentage changes as a pandas Series
    """

    quarterly_percentage_changes = (1 + annual_percentage_changes)/100 ** (1/4) - 1

    return quarterly_percentage_changes
