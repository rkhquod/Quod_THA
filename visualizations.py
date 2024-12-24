"""
Contains functions for exploratory analysis and plots:
1) Transactions per customer
2) Monthly frequency of a specific product in 2018
3) Top 5 products in the last 6 months
4) Time-series visualization for potential seasonality
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_transactions_by_customer(df: pd.DataFrame):
    """
    Create an ordered bar chart of total transactions per customer (descending).
    """
    transactions_per_customer = df['customer_id'].value_counts()

    plt.figure(figsize=(10, 6))
    transactions_per_customer.plot(kind='bar', color='blue')
    plt.title("Total Number of Transactions per Customer (Descending)")
    plt.xlabel("Customer ID")
    plt.ylabel("Transaction Count")
    plt.xticks([]) 
    plt.tight_layout()
    plt.show()


def plot_product_freq_for_2018(df: pd.DataFrame, product_id: str):
    """
    Given a product ID, plot its monthly transaction frequency for the year 2018.
    """
    df_2018 = df[df['date'].dt.year == 2018]
    df_product = df_2018[df_2018['product_id'] == product_id]

    monthly_counts = df_product['date'].dt.month.value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    monthly_counts.plot(kind='bar', color='green')
    plt.title(f"Monthly Transaction Frequency for '{product_id}' (2018)")
    plt.xlabel("Month")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.show()


def plot_top_5_products_last_6_months(df: pd.DataFrame):
    """
    Plot the top 5 products that drove the highest sales over the last 6 months.
    """
    max_date = df['date'].max()
    six_months_ago = max_date - pd.DateOffset(months=6)

    df_last_6 = df[df['date'] > six_months_ago]
    top_products = df_last_6['product_id'].value_counts().head(5)

    plt.figure(figsize=(8, 5))
    top_products.plot(kind='bar', color='orange')
    plt.title("Top 5 Products (Last 6 Months)")
    plt.xlabel("Product ID")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.show()


def plot_monthly_aggregate(df: pd.DataFrame):
    """
    Check for seasonality: plot the monthly total transactions (all customers, all products).
    """
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_totals = df.groupby('year_month').size()

    # Convert PeriodIndex to Timestamp for continuous plotting
    monthly_totals.index = monthly_totals.index.to_timestamp()

    plt.figure(figsize=(10, 5))
    monthly_totals.plot(marker='o')
    plt.title("Monthly Total Transactions (All Customers, All Products)")
    plt.xlabel("Year-Month")
    plt.ylabel("Transaction Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
