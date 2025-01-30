import pandas as pd
from tqdm import tqdm
from src.utils.constants import RAW_DATA_PATHS, TARGET_COLUMN

def load_raw_data(logger=None):
    
    if logger:
        logger.info("Loading raw data...")
    raw_dataframes = []
    for path in RAW_DATA_PATHS:
        df = pd.read_csv(path)
        if logger:
            logger.info(f"Loaded raw data from {path}. Shape: {df.shape}")
        raw_dataframes.append(df)
        
    return raw_dataframes

def clean_data(transaction_dataframes):
    """Concatenate and clean transaction data."""
    all_transactions = pd.concat(transaction_dataframes, axis=0).sort_values(by=["date"])
    all_transactions = all_transactions.drop_duplicates()
    all_transactions['date'] = pd.to_datetime(all_transactions['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
    return all_transactions

def generate_monthly_data(df, monthly_col="year_month", fill_value=0):
    """Generate monthly transaction data."""
    df[monthly_col] = df['date'].dt.to_period('M')
    periodic_data = df.groupby(['customer_id', monthly_col])['product_id'].count().reset_index().rename(
        columns={"product_id": "transactions"}
    )
    
    all_dates = pd.period_range(start=periodic_data[monthly_col].min(),
                                end=periodic_data[monthly_col].max(),
                                freq='M')
    
    all_combinations = pd.MultiIndex.from_product(
        [periodic_data['customer_id'].unique(), all_dates],
        names=['customer_id', monthly_col]
    ).to_frame(index=False)
    
    complete_data = all_combinations.merge(periodic_data, on=['customer_id', monthly_col], how='left')
    complete_data[TARGET_COLUMN] = complete_data['transactions'].fillna(fill_value)
    
    return complete_data

def feat_eng_monthly(monthly_transactions, min_lag=1, max_lag=13, window=1, fill_value=0.0):
    """Feature engineering for monthly data."""
    added_cols = []
    for lag in range(min_lag, max_lag, window):
        monthly_transactions[f'1_month_transactions_lag_{lag}'] = monthly_transactions.groupby('customer_id')['transactions'].shift(lag).fillna(fill_value)
        added_cols.append(f'1_month_transactions_lag_{lag}')
    return monthly_transactions, added_cols

def feat_eng_tri_monthly(monthly_transactions, min_lag=3, max_lag=13, window=3, fill_value=0.0):
    """Feature engineering for tri-monthly data."""
    added_cols = []
    for lag in range(min_lag, max_lag, window):
        monthly_transactions[f'3_month_transactions_lag_{lag}'] = monthly_transactions.groupby('customer_id')['transactions'].shift(lag).fillna(fill_value)
        added_cols.append(f'3_month_transactions_lag_{lag}')
    return monthly_transactions, added_cols


def tri_monthly_rolling_transaction_sums(monthly_data, window_size=3):
    # Sort the data by customer and date
    monthly_data = monthly_data.sort_values(by=['customer_id', 'year_month'])

    def calc_rolling_sums(group):
        
        group[TARGET_COLUMN] = group[TARGET_COLUMN].shift(-(window_size - 1)).rolling(window=window_size, min_periods=1).sum()
        # Extract start and end months for each rolling window
        group['start_month'] = group['year_month'].dt.month
        group['start_year'] = group['year_month'].dt.year
        
        group['end_month'] = group['start_month'] + window_size - 1
        group['end_year'] = group['start_year']
        
        overflow = group['end_month'] > 12
        group.loc[overflow, 'end_month'] -= 12
        group.loc[overflow, 'end_year'] += 1
        
        group = group.dropna(subset=[TARGET_COLUMN])
        return group

    # Apply the calculation group by customer
    monthly_data = monthly_data.groupby('customer_id', group_keys=False).apply(calc_rolling_sums)
    
    # Sort the final result
    monthly_data = monthly_data.sort_values(by=['customer_id', 'start_year', 'start_month'])
    monthly_data = monthly_data.drop(columns = ["year_month"])
    
    return monthly_data


def train_test_split(transaction_data, train_year=2019, train_month=1):
    """Split data into train and test sets."""
    if train_month == 12:
        test_year = train_year + 1
        test_month = 1
    else:
        test_year = train_year
        test_month = train_month + 1

    # Filter training data
    train = transaction_data[
        (transaction_data['end_year'] < train_year) |
        ((transaction_data['end_year'] == train_year) & (transaction_data['end_month'] <= train_month))
    ]

    # Filter testing data
    test = transaction_data[
        (transaction_data['start_year'] == test_year) &
        (transaction_data['start_month'] == test_month)
    ]
    
    return train, test