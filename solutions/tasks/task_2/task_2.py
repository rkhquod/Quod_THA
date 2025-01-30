import matplotlib.pyplot as plt
import pandas as pd

from solutions.utils import load_data, save_results_plot


def plot_transaction_frequency(product_id: str, df: pd.DataFrame, year: int = 2018) -> None:
    """
    Plots the transaction frequency per month for a given product ID and year.

    Args:
        product_id (str): The ID of the product for which the transaction frequency will be plotted.
        df (pd.DataFrame): The input DataFrame containing transaction data.
            Must include columns: "date" (datetime), "product_id" (str).
        year (int, optional): The year for which the transaction frequency is plotted.
            Defaults to 2018.

    Returns:
        None: The function generates and saves a plot but does not return any value.
    """
    # Filter data
    df = df[df["date"].dt.year == year]

    # Filter data for the given product ID
    product_data = df[df["product_id"] == product_id]

    # Extract year and month, and count transactions
    product_data["year_month"] = product_data["date"].dt.to_period("M")
    monthly_transactions = product_data.groupby("year_month").size().reset_index(name="transaction_count")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_transactions["year_month"].astype(str), monthly_transactions["transaction_count"], marker="o",
             color="skyblue")
    plt.title(f"Transaction Frequency per Month for Product ID: {product_id} ({year})")
    plt.xlabel("Month")
    plt.ylabel("Transaction Count")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    save_results_plot(n_task=2, plot_name='transactions_frequency')


PRODUCT_ID = "Volkswagen"

if __name__ == '__main__':
    # Load data
    df_data = load_data()

    # Make plot
    plot_transaction_frequency(product_id=PRODUCT_ID, df=df_data, year=2018)
