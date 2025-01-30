import os
import pandas as pd
import numpy as np
import torch
from joblib import load

from solutions.tasks.task_3.lstm_model import LSTMForecast, multi_step_forecast
from solutions.utils import load_data


def predict_for_customer(customer_id, model_type, train_end_date, test_start_date, test_end_date, daily_per_customer):
    """
    Loads the appropriate model from disk and performs a prediction for a given customer,
    returning the sum of predictions and the sum of actual transactions in the test period.
    """

    # Customer data
    df_cust = daily_per_customer.loc[
        daily_per_customer['customer_id'] == customer_id,
        ['date', 'transaction_count']
    ].copy()
    if df_cust.empty:
        return None, None

    df_cust.set_index('date', inplace=True)
    df_cust.sort_index(inplace=True)

    # fill in the missing days with zeros
    full_range = pd.date_range(df_cust.index.min(), test_end_date, freq='D')
    df_cust_filled = df_cust.reindex(full_range, fill_value=0)

    # Train/test split
    train_data = df_cust_filled.loc[:train_end_date]
    test_data = df_cust_filled.loc[test_start_date:test_end_date]

    if len(train_data) == 0 or len(test_data) == 0:
        return None, None

    # Model path
    if model_type == "LSTM":
        # "LSTM_customer_{id}.pt"
        model_path = f"models/LSTM_customer_{customer_id}.pt"
    else:
        # ARIMA -> "ARIMA_customer_{id}.pkl", LGBM -> "LGBM_customer_{id}.pkl"
        model_path = f"models/{model_type}_customer_{customer_id}.pkl"

    if not os.path.exists(model_path):
        # Nie mamy wytrenowanego modelu
        return None, None

    # -----------------------------------------------------
    # Choose model type
    # -----------------------------------------------------
    if model_type == "ARIMA":
        loaded_model = load(model_path)  # ARIMAResults
        forecast_steps = len(test_data)
        y_pred = loaded_model.forecast(steps=forecast_steps).values
        y_true = test_data['transaction_count'].values

    elif model_type == "LGBM":
        loaded_model = load(model_path)  # LGBMRegressor

        # Recreate features (lags) – the same as in training.py
        df_lags = pd.DataFrame(df_cust_filled)
        df_lags.columns = ['y']

        max_lag = 7
        for lag in range(1, max_lag + 1):
            df_lags[f'lag_{lag}'] = df_lags['y'].shift(lag)
        df_lags.dropna(inplace=True)

        df_test = df_lags.loc[test_start_date:test_end_date]
        if len(df_test) == 0:
            return None, None

        X_test = df_test.drop(columns=['y'])
        y_true = df_test['y'].values

        y_pred = loaded_model.predict(X_test)

    elif model_type == "LSTM":
        # Inference LSTM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a model object in the same configuration as in training
        model_lstm = LSTMForecast(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2)
        model_lstm.load_state_dict(torch.load(model_path, map_location=device))
        model_lstm.to(device)

        # Preparing data for prediction
        seq_length = 7  # Like in training
        arr_cust = df_cust_filled['transaction_count'].values.astype(float)

        # Take the last seq_length days from train for the forecast
        arr_train = arr_cust[df_cust_filled.index <= train_end_date]
        arr_test = arr_cust[
            (df_cust_filled.index >= test_start_date) &
            (df_cust_filled.index <= test_end_date)
        ]
        if len(arr_test) == 0:
            return None, None

        if len(arr_train) < seq_length:
            return None, None

        last_seq = arr_train[-seq_length:]  # last 7 days to train
        forecast_horizon = len(arr_test)    # days to predict

        # Iterative multi-step forecast
        y_pred = multi_step_forecast(model_lstm,
                                     init_seq=last_seq,
                                     forecast_horizon=forecast_horizon,
                                     device=device)
        y_true = arr_test

    else:
        return None, None

    return np.sum(y_true), np.sum(y_pred)


if __name__ == "__main__":
    # Compare with the same dates as in training.py
    train_end_date = pd.Timestamp("2019-01-31")
    test_start_date = pd.Timestamp("2019-02-01")
    test_end_date = pd.Timestamp("2019-04-30")

    # Load data
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])

    daily_per_customer = (
        data.groupby(['customer_id', data['date'].dt.date])
        .size()
        .reset_index(name='transaction_count')
    )
    daily_per_customer['date'] = pd.to_datetime(daily_per_customer['date'])

    # Choose model to inference
    model_type = input("Which model do you want to evaluate? (ARIMA/LGBM/LSTM): ").strip().upper()
    if model_type not in ["ARIMA", "LGBM", "LSTM"]:
        print("Unknown model type. Exiting.")
        exit()

    unique_customers = daily_per_customer['customer_id'].unique()

    # Variables for accumulating errors
    abs_errors = []
    sq_errors = []

    for i, customer_id in enumerate(unique_customers):
        y_true_sum, y_pred_sum = predict_for_customer(
            customer_id,
            model_type,
            train_end_date,
            test_start_date,
            test_end_date,
            daily_per_customer
        )
        if y_true_sum is None or y_pred_sum is None:
            continue

        error = y_pred_sum - y_true_sum
        abs_errors.append(abs(error))
        sq_errors.append(error ** 2)

    if len(abs_errors) == 0:
        print(f"No valid predictions for any customer with model '{model_type}'.")
    else:
        mean_ae = np.mean(abs_errors)
        mean_se = np.mean(sq_errors)
        mean_mse = mean_se
        mean_rmse = np.sqrt(mean_mse)

        print(f"\n=== Inference with {model_type} model ===")
        print(f"Mean AE:   {mean_ae:.2f}")
        print(f"Mean SE:   {mean_se:.2f}")
        print(f"Mean MSE:  {mean_mse:.2f}")
        print(f"Mean RMSE: {mean_rmse:.2f}")
