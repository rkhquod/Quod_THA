from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.utils import get_top_products_last_6_months
from src.utils.constants import OUTPUT_FOLDER
from src.preprocessing.utils import load_raw_data, clean_data
from src.utils.logger import setup_logger  
import os

fig_size = (15, 7)

def plot_count_per_product(df) :
    products =  Counter(df["product_id"])
    sorted_products = dict(sorted(list(products.items()), key = lambda x : x[1], reverse = True))
    fig = plt.figure(figsize=fig_size)
    plt.bar(sorted_products.keys(), sorted_products.values());
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("company")
    plt.ylabel("Count")
    plt.title("Total count of transactions for each product", 
              fontsize=20)
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"customer_count_per_product.png"),
        bbox_inches='tight',
        pad_inches=0.1,  
    )
    plt.show()
    
def plot_product_count(df, agg_fct = "count", ascending = False) : 
    """Function to generate plot of number of transactions per customer 
    
    Args:
        df (pd.DataFrame): Dataframe containing transactions per customer
        ascending (bool, optional): sorted by ascending order. Defaults to False.
        agg_fct (str, optional): aggregation function. Defaults to "count".
        size (tuple, optional): size of figure. Defaults to (20, 7).
    """
    df_per_customer_per_period = df[["customer_id", "product_id"]].groupby(["customer_id"], axis = 0).agg(agg_fct)
    cutomer_to_count = df_per_customer_per_period.sort_values(by = "product_id", ascending = ascending).to_dict()["product_id"]
    customer_idx, counts = [i for i in range(len(cutomer_to_count.keys()))], cutomer_to_count.values()
    fig = plt.figure(figsize = fig_size)
    plt.bar(customer_idx, counts);
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Customer")
    plt.ylabel(agg_fct)
    plt.title(f"{agg_fct} of transactions for each customer", fontsize=20)
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"product_count_per_customer.png"),
        bbox_inches='tight',
        pad_inches=0.1,  
    )
    plt.show()
    
def plot_product_count_per_customer(df, product_name, granularity = "year_month", size = (20, 14)) :
    """Function to generate plot of number of customers for each product
    It also plots the top 2 customers transactions on this product.
    The period is from 2017 to first quarted of 2020

    Args:
        df (pd.DataFrame): Dataframe containing transactions per customer
        product_name (str, optional): name of the product.
        size (tuple, optional): size of figure. Defaults to (20, 7).
    """
    df[granularity] = df['date'].dt.to_period('M')
    date_to_count=df[df["product_id"] == product_name][[granularity, "customer_id"]].groupby([granularity], axis = 0).agg("count").to_dict()["customer_id"]
    dates, counts = date_to_count.keys(), date_to_count.values()
    dates = [period.to_timestamp() for period in dates]
    
    fig = plt.figure(figsize = fig_size)
    plt.bar(dates, counts, color='skyblue')
    plt.plot(dates, counts)
    plt.xticks(dates)
    plt.tick_params(axis='x', rotation=90)
    plt.xlabel("Date")
    plt.ylabel("count")
    plt.title(f"Monthly count of transactions for {product_name}")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"monthly_{product_name}_transactions.png"),
        bbox_inches='tight',
        pad_inches=0.1,  
    )
    plt.show()


def plot_seasonality(transactions):
    top_products_per_period = get_top_products_last_6_months(transactions)
    top_products_per_period.index = pd.to_datetime(top_products_per_period.index)
    years = top_products_per_period.index.year.unique()
    min_date = top_products_per_period.index.min()
    max_date = top_products_per_period.index.max()
    
    fig = plt.figure(figsize = fig_size)
    
    for product_id in top_products_per_period.columns: 
        plt.plot(top_products_per_period.index, top_products_per_period[product_id], marker='o', label=f"{product_id}")
    for year in years:
        start_of_year = pd.to_datetime(f'{year}-01-01')
        if start_of_year >= min_date and start_of_year <= max_date:
            plt.axvline(x=start_of_year, color='black', linestyle='-', linewidth=2)
        for i in range(1, 5):  # There are 4 quarters in a year
            quarter_start = start_of_year + pd.DateOffset(months=3 * i)
            if quarter_start >= min_date and quarter_start <= max_date:
                plt.axvline(x=quarter_start, color='g', linestyle='-.', linewidth=1)

    plt.title("Total sales in the last 6 months per product", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales Count", fontsize=12)
    plt.legend(title="Product ID", loc='upper left')

    plt.grid()
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"top_products.png"),
        bbox_inches='tight',
        pad_inches=0.1,
    )
    plt.show()
    
def main() :
    logger = setup_logger("viz", "artifacts/visualization", save_to_file=True)
    
    df = load_raw_data(logger)
    df = clean_data(df)
    plot_count_per_product(df)
    plot_product_count_per_customer(df, product_name="Peugeot")
    plot_product_count(df)
    plot_seasonality(df)
    
    logger.info("All plots have been generated and saved.")
    
