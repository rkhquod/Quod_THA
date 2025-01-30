from joblib import dump
import os
from typing import Tuple

from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

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
    model_type = input("Which model do you want to use? (ARIMA/LGBM): ").strip().upper()
    if model_type not in ["ARIMA", "LGBM"]:
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
        test_end_date = train_end_date + pd.Timedelta(days=90)  # ~ 3 miesiące

    # --------------------------------------
    # 3) Loading and preparing data
    # --------------------------------------
    data = load_data()  # Expect columns: ['customer_id', 'date', 'product', ...]
    data['date'] = pd.to_datetime(data['date'])

    # Group: (customer, date) -> number of transactions on a given day
    daily_per_customer = (
        data.groupby(['customer_id', data['date'].dt.date])
        .size()
        .reset_index(name='transaction_count')
    )
    daily_per_customer['date'] = pd.to_datetime(daily_per_customer['date'])

    # Lists for storing error metrics (for the entire set of clients)
    abs_errors = []
    sq_errors = []

    # Prepare the models/ folder
    os.makedirs("models", exist_ok=True)

    # For simplicity - in this example we save a **separate** model per client.
    # In practice, you can also keep everything in one dictionary and dump it to one file.
    unique_customers = daily_per_customer['customer_id'].unique()

    for i, customer_id in enumerate(unique_customers):
        print(f'{i + 1}/{len(unique_customers)}: customer_id = {customer_id}')

        # Filter data for a single client
        df_cust = daily_per_customer.loc[
            daily_per_customer['customer_id'] == customer_id,
            ['date', 'transaction_count']
        ].copy()
        df_cust.set_index('date', inplace=True)
        df_cust.sort_index(inplace=True)

        # Fill in the missing days with zeros
        full_range = pd.date_range(df_cust.index.min(), test_end_date, freq='D')
        df_cust_filled = df_cust.reindex(full_range, fill_value=0)

        # Division into training and test sets
        train_data = df_cust_filled.loc[:train_end_date]
        test_data = df_cust_filled.loc[test_start_date:test_end_date]

        if len(train_data) == 0:
            # If not data -> skip it.
            continue

        # ------------------------------------------
        # Model training
        # ------------------------------------------
        if model_type == "ARIMA":
            try:
                # Build an ARIMA(2,1,2) model – example. These parameters can be chosen differentaly.
                arima_model = ARIMA(train_data['transaction_count'], order=(2, 1, 2))
                arima_result = arima_model.fit()

                # Forecast for number of days in test_data
                forecast_steps = len(test_data)
                forecast_result = arima_result.forecast(steps=forecast_steps)

                # y_pred and y_true – number of transactions on each test day
                y_pred = forecast_result.values
                y_true = test_data['transaction_count'].values

                # Save the trained model to a file (e.g. "models/ARIMA_123.pkl" for customer_id=123)
                dump(arima_result, f"models/{model_type}_customer_{customer_id}.pkl")

            except Exception as e:
                print(f"ARIMA training failed for customer {customer_id}, skipping. Error: {e}")
                continue

        else:
            # Build an LGBM model with simple features based on lags from the previous 7 days
            df_lags = pd.DataFrame(df_cust_filled)
            df_lags.columns = ['y']

            max_lag = 7
            for lag in range(1, max_lag + 1):
                df_lags[f'lag_{lag}'] = df_lags['y'].shift(lag)

            # Remove rows with NaN (the beginning of the series where we have no lags)
            df_lags.dropna(inplace=True)

            # Train/test split based on index (i.e. date)
            df_train = df_lags.loc[:train_end_date]
            df_test = df_lags.loc[test_start_date:test_end_date]

            # If there is no data -> skip it.
            if len(df_train) == 0 or len(df_test) == 0:
                continue

            # X is lag features, y is the value from that day
            X_train = df_train.drop(columns=['y'])
            y_train = df_train['y']
            X_test = df_test.drop(columns=['y'])
            y_test = df_test['y']

            # Model training
            if len(X_train) < 2:
                # To less data
                continue

            model_lgbm = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            model_lgbm.fit(X_train, y_train)

            # Prediction
            y_pred = model_lgbm.predict(X_test)
            y_true = y_test.values

            # Save the trained model
            dump(model_lgbm, f"models/{model_type}_customer_{customer_id}.pkl")

        # ------------------------------------------
        # Count transaction totals over the test horizon (3 months)
        # ------------------------------------------
        y_true_sum = np.sum(y_true)
        y_pred_sum = np.sum(y_pred)

        ae, se, _ = evaluate_3m_forecast(y_true_sum, y_pred_sum)
        abs_errors.append(ae)
        sq_errors.append(se)

    # --------------------------------------
    # 4) Overall metrics
    # --------------------------------------
    if len(abs_errors) > 0:
        mean_ae = np.mean(abs_errors)  # Mean Absolute Error (for 3-months sum)
        mean_se = np.mean(sq_errors)
        mean_mse = mean_se  # In this case - this is same
        mean_rmse = np.sqrt(mean_mse)

        print(f"=== Overall {model_type} performance (3-month totals) ===")
        print(f"Mean AE:   {mean_ae:.2f}")
        print(f"Mean SE:   {mean_se:.2f}")
        print(f"Mean MSE:  {mean_mse:.2f}")
        print(f"Mean RMSE: {mean_rmse:.2f}")
    else:
        print("No valid forecasts generated.")
