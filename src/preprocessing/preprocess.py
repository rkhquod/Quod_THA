# scripts/preprocess.py
import pandas as pd
from pathlib import Path
from src.preprocessing.utils import (
    clean_data, generate_monthly_data, feat_eng_monthly,
    tri_monthly_rolling_transaction_sums, feat_eng_tri_monthly, train_test_split
)
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("engineered_features", "artifacts/preprocessing", save_to_file=True)
    
    def generate_monthly_data(self, cleaned_data):
        self.logger.info("Generating monthly data...")
        monthly_data = generate_monthly_data(
            cleaned_data,
            periodic_col=self.config["preprocessing"]["monthly_data"]["periodic_col"],
            fill_value=self.config["preprocessing"]["monthly_data"]["fill_value"]
        )
        self.logger.info(f"Monthly data generated. Shape: {monthly_data.shape}")
        return monthly_data
    
    def feat_eng_monthly(self, monthly_data):
        self.logger.info("Feature engineering for monthly data...")
        monthly_data, eng_cols = feat_eng_monthly(
            monthly_data,
            min_lag=self.config["preprocessing"]["feature_engineering"]["monthly"]["min_lag"],
            max_lag=self.config["preprocessing"]["feature_engineering"]["monthly"]["max_lag"],
            window=self.config["preprocessing"]["feature_engineering"]["monthly"]["window"],
            fill_value=self.config["preprocessing"]["feature_engineering"]["monthly"]["fill_value"]
        )
        self.logger.info(f"Monthly feature engineering completed. Shape: {monthly_data.shape}")
        self.logger.info(f"Columns added: {sorted(eng_cols)}")
        return monthly_data
    
    def tri_monthly_rolling_transaction_sums(self, monthly_data):
        self.logger.info("Calculating rolling transaction sums...")
        rolling_data = tri_monthly_rolling_transaction_sums(
            monthly_data,
            window_size=self.config["preprocessing"]["rolling_window"]["window_size"]
        )
        self.logger.info(f"Rolling transaction sums calculated. Shape: {rolling_data.shape}")
        return rolling_data
    
    def feat_eng_tri_monthly(self, rolling_data):
        self.logger.info("Feature engineering for tri-monthly data...")
        tri_monthly_data, eng_cols = feat_eng_tri_monthly(
            rolling_data,
            min_lag=self.config["preprocessing"]["feature_engineering"]["tri_monthly"]["min_lag"],
            max_lag=self.config["preprocessing"]["feature_engineering"]["tri_monthly"]["max_lag"],
            window=self.config["preprocessing"]["feature_engineering"]["tri_monthly"]["window"],
            fill_value=self.config["preprocessing"]["feature_engineering"]["tri_monthly"]["fill_value"]
        )
        self.logger.info(f"Tri-monthly feature engineering completed. Shape: {tri_monthly_data.shape}")
        self.logger.info(f"Columns added: {sorted(eng_cols)}")
        return tri_monthly_data


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("process", "artifacts/preprocessing", save_to_file=True)
        self.feature_engineer = FeatureEngineer(config)
    
    def load_raw_data(self):
        self.logger.info("Loading raw data...")
        raw_data_paths = self.config["data"]["raw_data_paths"]
        raw_dataframes = [pd.read_csv(path) for path in raw_data_paths]
        for i, df in enumerate(raw_dataframes):
            self.logger.info(f"Loaded raw data from {raw_data_paths[i]}. Shape: {df.shape}")
        return raw_dataframes
    
    def clean_data(self, transaction_dataframes):
        self.logger.info("Cleaning data...")
        cleaned_data = clean_data(transaction_dataframes)
        self.logger.info(f"Data cleaned. Shape: {cleaned_data.shape}")
        return cleaned_data
    
    def train_test_split(self, rolling_data):
        self.logger.info("Splitting data into train and test sets...")
        train_data, test_data = train_test_split(
            rolling_data,
            train_year=self.config["data"]["train_test_split"]["train_year"],
            train_month=self.config["data"]["train_test_split"]["train_month"]
        )
        self.logger.info(f"Train-test split completed. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        return train_data, test_data

    def preprocess_data(self):
        self.logger.info("Starting preprocessing pipeline...")
        
        raw_dataframes = self.load_raw_data()
        cleaned_data = self.clean_data(raw_dataframes)
        
        monthly_data = self.feature_engineer.generate_monthly_data(cleaned_data)
        monthly_data = self.feature_engineer.feat_eng_monthly(monthly_data)
        
        rolling_data = self.feature_engineer.tri_monthly_rolling_transaction_sums(monthly_data)
        rolling_data = self.feature_engineer.feat_eng_tri_monthly(rolling_data)
        
        train_data, test_data = self.train_test_split(rolling_data)
        
        processed_data_dir = Path(self.config["data"]["processed_data_dir"])
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        train_data.to_csv(processed_data_dir / "preprocessed_train.csv", index=False)
        test_data.to_csv(processed_data_dir / "preprocessed_test.csv", index=False)
        
        self.logger.info(f"Preprocessed data saved to {processed_data_dir}")


def main():
    config = load_config("configs/preprocess_config.yaml")
    preprocessor = Preprocessor(config)
    preprocessor.preprocess_data()


if __name__ == "__main__":
    main()
