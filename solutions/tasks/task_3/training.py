from joblib import dump
import os
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from lightgbm import LGBMRegressor

import torch

from solutions.tasks.task_3.lstm_model import (
    LSTMForecast,
    create_data_loader,
    train_lstm,
    multi_step_forecast
)
from solutions.utils import load_data


def evaluate_3m_forecast(y_true_sum: float, y_pred_sum: float) -> Tuple[float, float, float]:
    """
    Calculates errors for a 3-month forecast for a single client.

    Args:
        y_true_sum (float): The actual total value over three months.
        y_pred_sum (float): The predicted total value over three months.

    Returns:
        Tuple[float, float, float]:
            - Absolute Error (AE): The absolute difference between predicted and actual values.
            - Squared Error (SE): The squared difference between predicted and actual values.
            - Error: The raw difference between predicted and actual values.
    """
    error = y_pred_sum - y_true_sum
    abs_error = abs(error)
    sq_error = error ** 2
    return abs_error, sq_error, error


if __name__ == "__main__":
    # --------------------------------------
    # 1) Model selection
    # --------------------------------------
    model_type = input("Which model do you want to use? (ARIMA/LGBM/LSTM): ").strip().upper()
    if model_type not in ["ARIMA", "LGBM", "LSTM"]:
        print("Unknown model type. Defaulting to ARIMA.")
        model_type = "ARIMA"

    # --------------------------------------
    # 2) Training/Testing Period
    # --------------------------------------
    use_sample_dates = input("Do you want to use sample dates (Y/N)? ").strip().upper()
    if use_sample_dates == "Y":
        train_end_date = pd.Timestamp("2019-01-31")
        test_start_date = pd.Timestamp("2019-02-01")
        test_end_date = pd.Timestamp("2019-04-30")
    else:
        print("Please enter dates in format YYYY-MM-DD.")
        train_end_date_str = input("Enter the train end date (e.g., 2019-01-31): ")
        train_end_date = pd.Timestamp(train_end_date_str)
        test_start_date = train_end_date + pd.Timedelta(days=1)
        test_end_date = train_end_date + pd.Timedelta(days=90)

    # --------------------------------------
    # 3) Loading and preparing data
    # --------------------------------------
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])

    # Group by client and date -> we count the number of transactions on a given day
    daily_per_customer = (
        data.groupby(['customer_id', data['date'].dt.date])
        .size()
        .reset_index(name='transaction_count')
    )
    daily_per_customer['date'] = pd.to_datetime(daily_per_customer['date'])

    # Make directory to models
    os.makedirs("models", exist_ok=True)

    # Lists to collect errors
    abs_errors = []
    sq_errors = []

    unique_customers = daily_per_customer['customer_id'].unique()

    for i, customer_id in enumerate(unique_customers):
        print(f'{i + 1}/{len(unique_customers)}: customer_id = {customer_id}')

        # Data for single customer
        df_cust = daily_per_customer.loc[
            daily_per_customer['customer_id'] == customer_id,
            ['date', 'transaction_count']
        ].copy()
        df_cust.set_index('date', inplace=True)
        df_cust.sort_index(inplace=True)

        if len(df_cust) == 0:
            continue
        # Fill empty data with zeros
        full_range = pd.date_range(df_cust.index.min(), test_end_date, freq='D')
        df_cust_filled = df_cust.reindex(full_range, fill_value=0)

        # split to train/test
        train_data = df_cust_filled.loc[:train_end_date]
        test_data = df_cust_filled.loc[test_start_date:test_end_date]

        if len(train_data) == 0:
            continue

        y_pred = None
        y_true = None

        # ------------------------------------------
        # MODELE
        # ------------------------------------------
        if model_type == "ARIMA":
            try:
                arima_model = ARIMA(train_data['transaction_count'], order=(2, 1, 2))
                arima_result = arima_model.fit()

                # Forecast for number of days in test_data
                forecast_steps = len(test_data)
                forecast_result = arima_result.forecast(steps=forecast_steps)

                y_pred = forecast_result.values
                y_true = test_data['transaction_count'].values

                dump(arima_result, f"models/{model_type}_customer_{customer_id}.pkl")
            except Exception as e:
                print(f"ARIMA training failed for customer {customer_id}, skipping. Error: {e}")
                continue

        elif model_type == "LGBM":
            df_lags = pd.DataFrame(df_cust_filled)
            df_lags.columns = ['y']

            max_lag = 7
            for lag in range(1, max_lag + 1):
                df_lags[f'lag_{lag}'] = df_lags['y'].shift(lag)

            df_lags.dropna(inplace=True)

            df_train = df_lags.loc[:train_end_date]
            df_test = df_lags.loc[test_start_date:test_end_date]

            if len(df_train) == 0 or len(df_test) == 0:
                continue

            X_train = df_train.drop(columns=['y'])
            y_train = df_train['y']
            X_test = df_test.drop(columns=['y'])
            y_test = df_test['y']

            if len(X_train) < 2:
                continue

            model_lgbm = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            model_lgbm.fit(X_train, y_train)

            y_pred = model_lgbm.predict(X_test)
            y_true = y_test.values

            dump(model_lgbm, f"models/{model_type}_customer_{customer_id}.pkl")

        elif model_type == "LSTM":
            # Make numpy array
            arr_cust = df_cust_filled['transaction_count'].values.astype(float)
            arr_train = arr_cust[df_cust_filled.index <= train_end_date]
            arr_test = arr_cust[
                (df_cust_filled.index >= test_start_date) &
                (df_cust_filled.index <= test_end_date)
            ]

            # Check if we have enough points to make sequences
            seq_length = 7
            if len(arr_train) <= (seq_length + 1):
                continue

            # Train/val split
            val_size = int(len(arr_train) * 0.1)
            if val_size < seq_length:
                val_size = seq_length

            train_array = arr_train[:-val_size]
            val_array = arr_train[-val_size:]

            # Make DataLoader
            batch_size = 16
            try:
                train_loader = create_data_loader(train_array,
                                                  seq_length=seq_length,
                                                  batch_size=batch_size,
                                                  shuffle=True)
            except ValueError:
                continue
            val_loader = create_data_loader(val_array,
                                            seq_length=seq_length,
                                            batch_size=batch_size,
                                            shuffle=False)

            # Get device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Model init
            model_lstm = LSTMForecast(input_dim=1,
                                      hidden_dim=32,
                                      num_layers=2,
                                      output_dim=1,
                                      dropout=0.2).to(device)

            # Train model
            train_lstm(model_lstm,
                       train_loader=train_loader,
                       val_loader=val_loader,
                       epochs=50,
                       lr=1e-3,
                       device=device)

            # Multi-step prediction
            # Take the last seq_length values from the entire train
            last_seq = arr_train[-seq_length:]
            forecast_horizon = len(arr_test)
            if forecast_horizon == 0:
                continue

            y_pred = multi_step_forecast(model_lstm,
                                         last_seq,
                                         forecast_horizon=forecast_horizon,
                                         device=device)
            y_true = arr_test

            # Save trained model
            model_path = f"models/LSTM_customer_{customer_id}.pt"
            torch.save(model_lstm.state_dict(), model_path)
            print(f"Saved LSTM model for customer {customer_id} -> {model_path}")

        else:
            raise ValueError(f"Unexpected model_type: {model_type}")

        # ------------------------------------------
        # Calculate errors for the 3-month total (Feb+Mar+Apr 2019)
        # ------------------------------------------
        if y_pred is not None and y_true is not None and len(y_true) > 0:
            y_true_sum = np.sum(y_true)
            y_pred_sum = np.sum(y_pred)

            ae, se, _ = evaluate_3m_forecast(y_true_sum, y_pred_sum)
            abs_errors.append(ae)
            sq_errors.append(se)

    # --------------------------------------
    # 4) Overall metrics
    # --------------------------------------
    if len(abs_errors) > 0:
        mean_ae = np.mean(abs_errors)
        mean_se = np.mean(sq_errors)
        mean_mse = mean_se  # In this case SE is already 'per client', but we average
        mean_rmse = np.sqrt(mean_mse)

        print(f"\n=== Overall {model_type} performance (3-month totals) ===")
        print(f"Mean AE:   {mean_ae:.2f}")
        print(f"Mean SE:   {mean_se:.2f}")
        print(f"Mean MSE:  {mean_mse:.2f}")
        print(f"Mean RMSE: {mean_rmse:.2f}")
    else:
        print("No valid forecasts generated.")
