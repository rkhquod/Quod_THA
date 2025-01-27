import matplotlib.pyplot as plt

from solutions.const import CUSTOMER_ID_COLNAME
from solutions.utils import load_data, save_results_plot

if __name__ == '__main__':
    # Load data
    df_data = load_data()

    # Group transactions per customer
    transactions_per_customer = df_data.groupby(CUSTOMER_ID_COLNAME).size().reset_index(name="transaction_count")
    transactions_per_customer.sort_values(by="transaction_count", ascending=False, inplace=True)

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.bar(transactions_per_customer[CUSTOMER_ID_COLNAME].astype(str), transactions_per_customer["transaction_count"],
            color="blue")
    plt.title("Total Transactions per Customer (Descending Order)")
    plt.xlabel("Customer ID")
    plt.ylabel("Transaction Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_results_plot(n_task=1, plot_name='transactions_per_customer')
