import os
from functools import wraps
import pandas as pd
import matplotlib.pyplot as plt

# Decorator to save plots
# def save_plot(output_dir="plots"):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # Create the output directory if it doesn't exist
#             os.makedirs(output_dir, exist_ok=True)
            
#             # Execute the function
#             result = func(*args, **kwargs)
            
#             # Save the plot if the function returns a figure or axes
#             plt.savefig(
#                 os.path.join(output_dir, f"{func.__name__}.png"),
#                 transparent=True, 
#                 bbox_inches='tight',
#                 pad_inches=0.1,  
#             )
#             print(f"Plot saved to {os.path.join(output_dir, f'{func.__name__}.png')}")
            
#             return result
#         return wrapper
#     return decorator

def get_top_products_last_6_months(df, window=6, sliding_window=3, top_k=5):
    
    df['date'] = pd.to_datetime(df['date'])
    # Add a sales count column (1 sale per row)
    df['sales'] = 1
    # Get the minimum and maximum dates in the dataset
    min_date = df['date'].min()
    max_date = df['date'].max()
    # Initialize an empty list to store results
    results = []
    top_products_set = set()  # To track the unique set of top products
    current_date = min_date + pd.DateOffset(months=window)
    while current_date <= max_date:
        six_months_ago = current_date - pd.DateOffset(months=window)
        window_data = df[(df['date'] > six_months_ago) & (df['date'] <= current_date)]
        # Aggregate sales by product
        product_sales = window_data.groupby('product_id')['sales'].sum().reset_index()
        # Rank products by sales
        product_sales['rank'] = product_sales['sales'].rank(method='dense', ascending=False)
        # Get the top products
        top_sales = product_sales[product_sales['rank'] <= top_k].copy()
        top_sales['date'] = current_date
        top_products_set.update(top_sales['product_id'].unique())
        
        all_products_in_window = product_sales[['product_id', 'sales']].copy()
        all_products_in_window['date'] = current_date
        results.append(all_products_in_window)
        current_date += pd.DateOffset(months=sliding_window)
    
    all_products_df = pd.concat(results, ignore_index=True)
    pivot_table = all_products_df.pivot(index='date', columns='product_id', values='sales')   
    # Keep only the unique products that appeared in the top 5 at any point
    pivot_table = pivot_table[list(top_products_set)]
    return pivot_table
