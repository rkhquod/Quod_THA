import matplotlib.pyplot as plt
import pandas as pd

from solutions.const import DATE_COLNAME, PRODUCT_ID_COLNAME
from solutions.utils import save_results_plot, load_data


def get_top_products_from_last_months(
        chosen_date: str,
        df: pd.DataFrame,
        n_top_products: int = 5,
        last_n_months: int = 6) -> pd.DataFrame:
    """
    Get the top products based on transaction count over the last specified number of months.

    Args:
        chosen_date (str): The reference date in 'YYYY-MM-DD' format. Data will be filtered up to this date.
        df (pd.DataFrame): The input dataframe containing transaction data. It must include columns
            specified by `DATE_COLNAME` and `PRODUCT_ID_COLNAME`.
        n_top_products (int, optional): The number of top products to retrieve. Defaults to 5.
        last_n_months (int, optional): The number of months to consider for filtering data. Defaults to 6.

    Returns:
        pd.DataFrame: A dataframe containing the top products and their respective transaction counts,
        sorted in descending order by transaction count.
    """
    chosen_date = pd.Timestamp(chosen_date).tz_localize('UTC')

    # Filter data for the last 6 months
    six_months_earlier = chosen_date - pd.DateOffset(months=6)
    df_data_last_six_months = df[(df[DATE_COLNAME] >= six_months_earlier) & (df["date"] <= chosen_date)]

    # Group by product and count transactions
    product_sales = df_data_last_six_months.groupby(PRODUCT_ID_COLNAME).size().reset_index(name="transaction_count")

    # Sort by transaction count in descending order
    top_products = product_sales.sort_values(by="transaction_count", ascending=False).head(n_top_products)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(top_products["product_id"], top_products["transaction_count"], color="skyblue")
    plt.title(f"Top {n_top_products} Products by Transactions in the Last 6 Months ({chosen_date.date()})")
    plt.xlabel("Product ID")
    plt.ylabel("Transaction Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_results_plot(n_task=4, plot_name=f'top_{n_top_products}_products_last_{last_n_months}_months')

    return top_products


if __name__ == '__main__':
    input_date = input("Enter the date (e.g., 2019-01-31): ")

    # Load data
    df_data = load_data()

    # Get top 5 products
    top_5_products = get_top_products_from_last_months(chosen_date=input_date, df=df_data, n_top_products=5)
    print(top_5_products)
