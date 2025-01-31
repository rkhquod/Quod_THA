import os

import matplotlib.pyplot as plt
import pandas as pd

from solutions.const import data_1_filepath, data_2_filepath, DATE_COLNAME, RESULTS_DIR


def load_data() -> pd.DataFrame:
    """
    Loads and concatenates data from two CSV files, converting date column to datetime.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data from `data_1_filepath`
        and `data_2_filepath` with the date column converted to datetime format.
    """
    df_data_1 = pd.read_csv(data_1_filepath, index_col=0)
    df_data_2 = pd.read_csv(data_2_filepath, index_col=0)
    df_data = pd.concat([df_data_1, df_data_2])
    df_data[DATE_COLNAME] = pd.to_datetime(df_data[DATE_COLNAME])
    df_data = df_data.loc[~df_data['product_id'].isin(['Undefined', 'Not a make'])]

    return df_data


def save_results_plot(n_task: int, plot_name: str) -> None:
    """
    Saves a plot to the results directory under a task-specific folder.

    Args:
        n_task (int): The task number used to create the folder structure.
        plot_name (str): The name of the plot file (without extension).
    """
    plot_filepath = os.path.join(RESULTS_DIR, f'task_{n_task}', f'{plot_name}.png')
    os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
    plt.savefig(plot_filepath)
