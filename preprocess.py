# scripts/preprocess.py
import yaml
import pandas as pd
from pathlib import Path
from preprocessing_utils import (
    clean_data, generate_monthly_data, feat_eng_monthly,
    calculate_rolling_transaction_sums, feat_eng_tri_monthly, train_test_split
)

def load_config(config_path="preprocess_config.yaml"):
    """Load preprocessing configuration."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def preprocess_data(config):
    """Run the preprocessing pipeline."""
    # Load raw data
    raw_data_paths = config["data"]["raw_data_paths"]
    transaction_dataframes = [pd.read_csv(path) for path in raw_data_paths]
    
    # Clean data
    cleaned_data = clean_data(transaction_dataframes)
    
    # Generate monthly data
    monthly_data = generate_monthly_data(
        cleaned_data,
        periodic_col=config["preprocessing"]["monthly_data"]["periodic_col"],
        fill_value=config["preprocessing"]["monthly_data"]["fill_value"]
    )
    
    # Feature engineering for monthly data
    monthly_data = feat_eng_monthly(
        monthly_data,
        min_lag=config["preprocessing"]["feature_engineering"]["monthly"]["min_lag"],
        max_lag=config["preprocessing"]["feature_engineering"]["monthly"]["max_lag"],
        window=config["preprocessing"]["feature_engineering"]["monthly"]["window"],
        fill_value=config["preprocessing"]["feature_engineering"]["monthly"]["fill_value"]
    )
    
    # Calculate rolling transaction sums
    rolling_data = calculate_rolling_transaction_sums(
        monthly_data,
        window_size=config["preprocessing"]["rolling_window"]["window_size"]
    )
    
    # Feature engineering for tri-monthly data
    rolling_data = feat_eng_tri_monthly(
        rolling_data,
        min_lag=config["preprocessing"]["feature_engineering"]["tri_monthly"]["min_lag"],
        max_lag=config["preprocessing"]["feature_engineering"]["tri_monthly"]["max_lag"],
        window=config["preprocessing"]["feature_engineering"]["tri_monthly"]["window"],
        fill_value=config["preprocessing"]["feature_engineering"]["tri_monthly"]["fill_value"]
    )
    
    # Train-test split
    train_data, test_data = train_test_split(
        rolling_data,
        train_year=config["data"]["train_test_split"]["train_year"],
        train_month=config["data"]["train_test_split"]["train_month"]
    )
    print("Train data shape : ", train_data.shape)
    print("Test data shape : ", test_data.shape)
    
    # Save preprocessed data
    processed_data_dir = Path(config["data"]["processed_data_dir"])
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    train_data.to_csv(processed_data_dir / "preprocessed_train.csv", index=False)
    test_data.to_csv(processed_data_dir / "preprocessed_test.csv", index=False)
    
    print(f"Preprocessed data saved to {processed_data_dir}")

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)