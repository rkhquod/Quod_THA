import os
import pandas as pd
import numpy as np
from joblib import load
from statsmodels.tsa.arima.model import ARIMAResults
from lightgbm import LGBMRegressor

from solutions.utils import load_data


def predict_for_customer(customer_id, model_type, train_end_date, test_start_date, test_end_date, daily_per_customer):
    """
    Ładuje odpowiedni model z dysku i wykonuje predykcję dla zadanego klienta,
    zwracając sumę prognoz i sumę rzeczywistych transakcji w okresie testowym.
    """

    # Data per customer
    df_cust = daily_per_customer.loc[
        daily_per_customer['customer_id'] == customer_id,
        ['date', 'transaction_count']
    ].copy()
    df_cust.set_index('date', inplace=True)
    df_cust.sort_index(inplace=True)

    # Fill the missing days with zeros until the end of test_end_date
    full_range = pd.date_range(df_cust.index.min(), test_end_date, freq='D')
    df_cust_filled = df_cust.reindex(full_range, fill_value=0)

    # Divide data into training and test sets
    train_data = df_cust_filled.loc[:train_end_date]
    test_data = df_cust_filled.loc[test_start_date:test_end_date]

    if len(train_data) == 0 or len(test_data) == 0:
        # Return None if no data
        return None, None

    # Load model
    model_path = f"models/{model_type}_customer_{customer_id}.pkl"
    if not os.path.exists(model_path):
        # No model
        return None, None

    loaded_model = load(model_path)

    # Forecast depending on model type
    if model_type == "ARIMA":
        # Make sure that our training and testing objects are organized in the same way.
        forecast_steps = len(test_data)
        # forecast() must know how many steps ahead
        y_pred = loaded_model.forecast(steps=forecast_steps).values
        y_true = test_data['transaction_count'].values

    else:
        # LGBMRegressor
        # Recreate features (lags) – just like in main.py
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

        # Prediction / forecast
        y_pred = loaded_model.predict(X_test)

    # Return sums
    return np.sum(y_true), np.sum(y_pred)


if __name__ == "__main__":
    # Sample dates
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

    # Choose model to tests
    model_type = input("Which model do you want to evaluate? (ARIMA/LGBM): ").strip().upper()
    if model_type not in ["ARIMA", "LGBM"]:
        print("Unknown model type. Exiting.")
        exit()

    unique_customers = daily_per_customer['customer_id'].unique()

    # Variable for calculating total errors
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
            # No model or data
            continue

        # Liczymy błędy
        error = y_pred_sum - y_true_sum
        abs_errors.append(abs(error))
        sq_errors.append(error ** 2)

    if len(abs_errors) == 0:
        print("No valid predictions for any customer.")
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
