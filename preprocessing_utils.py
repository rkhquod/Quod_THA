# utils/preprocessing_utils.py
import pandas as pd
from tqdm import tqdm

def clean_data(transaction_dataframes):
    """Concatenate and clean transaction data."""
    all_transactions = pd.concat(transaction_dataframes, axis=0).sort_values(by=["date"])
    all_transactions = all_transactions.drop_duplicates()
    return all_transactions

def generate_monthly_data(df, periodic_col="year_month", fill_value=0):
    """Generate monthly transaction data."""
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
    df[periodic_col] = df['date'].dt.to_period('M')
    periodic_data = df.groupby(['customer_id', periodic_col])['product_id'].count().reset_index().rename(
        columns={"product_id": "transactions"}
    )
    
    all_dates = pd.period_range(start=periodic_data[periodic_col].min(),
                                end=periodic_data[periodic_col].max(),
                                freq='M')
    
    all_combinations = pd.MultiIndex.from_product(
        [periodic_data['customer_id'].unique(), all_dates],
        names=['customer_id', periodic_col]
    ).to_frame(index=False)
    
    complete_data = all_combinations.merge(periodic_data, on=['customer_id', periodic_col], how='left')
    complete_data['transactions'] = complete_data['transactions'].fillna(fill_value)
    
    return complete_data

def feat_eng_monthly(monthly_transactions, min_lag=1, max_lag=13, window=1, fill_value=0.0):
    """Feature engineering for monthly data."""
    for lag in range(min_lag, max_lag, window):
        monthly_transactions[f'1_month_transactions_lag_{lag}'] = monthly_transactions.groupby('customer_id')['transactions'].shift(lag).fillna(fill_value)
    return monthly_transactions

def calculate_rolling_transaction_sums(monthly_data, window_size=3):
    """Calculate rolling sums of transactions."""
    transformed_data = []
    
    for customer_id in tqdm(monthly_data['customer_id'].unique()):
        customer_data = monthly_data[monthly_data['customer_id'] == customer_id]
        
        for i in range(len(customer_data) - window_size + 1):
            target_sum = customer_data['transactions'].iloc[i:i+window_size].sum()
            lagged_transactions = monthly_data[monthly_data['customer_id'] == customer_id].iloc[i].drop(['transactions', 'year_month']).to_dict()
            
            transformed_data.append({
                'customer_id': customer_id,
                'start_year': customer_data['year_month'].iloc[i].year,
                'start_month': customer_data['year_month'].iloc[i].month,
                'end_year': customer_data['year_month'].iloc[i+window_size-1].year,
                'end_month': customer_data['year_month'].iloc[i+window_size-1].month,
                'transactions': target_sum,
                **lagged_transactions
            })
    
    transformed_data = pd.DataFrame(transformed_data)
    transformed_data = transformed_data.sort_values(by=['customer_id', 'start_year', 'start_month'])
    return transformed_data

def feat_eng_tri_monthly(monthly_transactions, min_lag=3, max_lag=13, window=3, fill_value=0.0):
    """Feature engineering for tri-monthly data."""
    for lag in range(min_lag, max_lag, window):
        monthly_transactions[f'3_month_transactions_lag_{lag}'] = monthly_transactions.groupby('customer_id')['transactions'].shift(lag).fillna(fill_value)
    return monthly_transactions

def train_test_split(transaction_data, train_year=2019, train_month=1):
    """Split data into train and test sets."""
    train = transaction_data[
        (transaction_data['end_year'] < train_year) |
        ((transaction_data['end_year'] == train_year) & (transaction_data['end_month'] <= train_month))
    ]
    
    test = transaction_data[
        (transaction_data['start_year'] == train_year) &
        (transaction_data['start_month'] == train_month + 1)
    ]
    
    return train, test