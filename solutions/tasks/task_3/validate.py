import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load
from typing import Optional

import altair as alt
import torch

from solutions.tasks.task_3.lstm_model import LSTMForecast, multi_step_forecast
from solutions.utils import load_data


def create_lag_features(df: pd.DataFrame, max_lag: int = 7) -> pd.DataFrame:
    """
    Creates columns lag_k for k in [1..max_lag], shifted relative to 'y'.
    Returns a DataFrame with all lags and the 'y' column.
    """
    df_lags = pd.DataFrame(df)
    df_lags.columns = ['y']

    for lag in range(1, max_lag + 1):
        df_lags[f'lag_{lag}'] = df_lags['y'].shift(lag)

    df_lags.dropna(inplace=True)
    return df_lags


def predict_daily_for_customer(customer_id: int,
                               model_type: str,
                               train_end_date: pd.Timestamp,
                               test_start_date: pd.Timestamp,
                               test_end_date: pd.Timestamp,
                               daily_per_customer: pd.DataFrame
                               ) -> Optional[pd.DataFrame]:
    """
    Loads a model from disk (in the 'models/' folder) and returns a DataFrame with columns:
    [date, actual, predicted] for the given customer's daily data in the test period.
    Returns None if no data or model is found.
    """
    # Filter data for this customer
    df_cust = daily_per_customer.loc[
        daily_per_customer['customer_id'] == customer_id,
        ['date', 'transaction_count']
    ].copy()
    if df_cust.empty:
        return None

    # Sort by date and fill missing days
    df_cust.set_index('date', inplace=True)
    df_cust.sort_index(inplace=True)

    full_range = pd.date_range(df_cust.index.min(), test_end_date, freq='D')
    df_cust_filled = df_cust.reindex(full_range, fill_value=0)

    # Split into train/test
    train_data = df_cust_filled.loc[:train_end_date]
    test_data = df_cust_filled.loc[test_start_date:test_end_date]

    if len(train_data) == 0 or len(test_data) == 0:
        return None

    # Set the path to the model – depending on the type:
    if model_type == "LSTM":
        model_path = f"models/LSTM_customer_{customer_id}.pt"
    else:
        model_path = f"models/{model_type}_customer_{customer_id}.pkl"

    if not os.path.exists(model_path):
        return None

    # ------------------------------------------------
    # 1) ARIMA
    # ------------------------------------------------
    if model_type == "ARIMA":
        loaded_model = load(model_path)  # ARIMAResults
        forecast_steps = len(test_data)
        forecast_result = loaded_model.forecast(steps=forecast_steps)

        df_predictions = pd.DataFrame({
            'date': test_data.index,
            'actual': test_data['transaction_count'].values,
            'predicted': forecast_result.values
        })
        df_predictions.set_index('date', inplace=True)
        return df_predictions

    # ------------------------------------------------
    # 2) LGBM
    # ------------------------------------------------
    elif model_type == "LGBM":
        loaded_model = load(model_path)  # LGBMRegressor
        df_lags = create_lag_features(df_cust_filled, max_lag=7)
        df_test = df_lags.loc[test_start_date:test_end_date]

        if df_test.empty:
            return None

        X_test = df_test.drop(columns=['y'])
        y_true = df_test['y'].values
        y_pred = loaded_model.predict(X_test)

        df_predictions = pd.DataFrame({
            'date': df_test.index,
            'actual': y_true,
            'predicted': y_pred
        })
        df_predictions.set_index('date', inplace=True)
        return df_predictions

    # ------------------------------------------------
    # 3) LSTM
    # ------------------------------------------------
    elif model_type == "LSTM":
        # Load the state of the trained LSTM model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initializing the model with the same configuration as in training.py
        model_lstm = LSTMForecast(input_dim=1,
                                  hidden_dim=32,
                                  num_layers=2,
                                  output_dim=1,
                                  dropout=0.2)
        model_lstm.load_state_dict(torch.load(model_path, map_location=device))
        model_lstm.to(device)
        model_lstm.eval()

        # Preparing data
        seq_length = 7
        arr_cust = df_cust_filled['transaction_count'].values.astype(float)

        arr_train = arr_cust[df_cust_filled.index <= train_end_date]
        arr_test = arr_cust[
            (df_cust_filled.index >= test_start_date) &
            (df_cust_filled.index <= test_end_date)
        ]

        if len(arr_test) == 0 or len(arr_train) < seq_length:
            return None

        # Last seq_length days from train to forecast start
        last_seq = arr_train[-seq_length:]
        forecast_horizon = len(arr_test)

        # Iterative forecast for the entire test period
        y_pred = multi_step_forecast(model_lstm,
                                     init_seq=last_seq,
                                     forecast_horizon=forecast_horizon,
                                     device=device)

        # Dataframe with results
        df_predictions = pd.DataFrame({
            'date': test_data.index,
            'actual': test_data['transaction_count'].values,
            'predicted': y_pred
        })
        df_predictions.set_index('date', inplace=True)
        return df_predictions

    return None


@st.cache_data(show_spinner=True)
def compute_validation(model_type: str,
                       train_end_date: pd.Timestamp,
                       test_start_date: pd.Timestamp,
                       test_end_date: pd.Timestamp,
                       daily_per_customer: pd.DataFrame,
                       customer_choice: str) -> pd.DataFrame:
    """
    Computes the validation results (3-month sum) for each customer.
    Returns a DataFrame with one row per customer: [customer_id, actual_3m_sum, pred_3m_sum, error].
    """
    unique_customers = daily_per_customer['customer_id'].unique()
    all_option = "All customers (aggregated errors)"

    # If a specific customer was selected, restrict the list
    if customer_choice != all_option:
        unique_customers = [customer_choice]

    abs_errors = []
    sq_errors = []
    results_for_customers = []

    # Iterate over each customer and forecast
    for customer_id in unique_customers:
        df_pred = predict_daily_for_customer(
            customer_id=customer_id,
            model_type=model_type,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            daily_per_customer=daily_per_customer
        )
        if df_pred is None or df_pred.empty:
            continue

        y_true_sum = df_pred['actual'].sum()
        y_pred_sum = df_pred['predicted'].sum()
        error = y_pred_sum - y_true_sum

        abs_errors.append(abs(error))
        sq_errors.append(error ** 2)

        results_for_customers.append({
            'customer_id': customer_id,
            'actual_3m_sum': y_true_sum,
            'pred_3m_sum': y_pred_sum,
            'error': error
        })

    if not results_for_customers:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_for_customers)
    return results_df


def main():
    st.title("ARIMA vs. LGBM vs. LSTM Model Validation – Forecasting Transactions")
    st.sidebar.header("Validation Settings")

    # Model type selection
    model_type = st.sidebar.selectbox(
        "Select model type:",
        ["ARIMA", "LGBM", "LSTM"],  # Dodajemy "LSTM"
        index=0
    )

    # Date input: default training end date / test period
    default_train_end = pd.Timestamp("2019-01-31")
    default_test_start = pd.Timestamp("2019-02-01")
    default_test_end = pd.Timestamp("2019-04-30")

    train_end_date = pd.Timestamp(default_train_end)
    test_start_date = pd.Timestamp(default_test_start)
    test_end_date = pd.Timestamp(default_test_end)

    # Load data
    data = load_data()
    data['date'] = pd.to_datetime(data['date'])
    # Remove timezone if present
    data["date"] = data["date"].dt.tz_localize(None)

    # Prepare daily data
    daily_per_customer = (
        data.groupby(['customer_id', data['date'].dt.date])
        .size()
        .reset_index(name='transaction_count')
    )
    daily_per_customer['date'] = pd.to_datetime(daily_per_customer['date'])

    # Customer selection (all vs. specific)
    unique_customers = daily_per_customer['customer_id'].unique()
    all_option = "All customers (aggregated errors)"
    customer_choice = st.sidebar.selectbox(
        "Select a customer for daily visualization:",
        options=[all_option] + list(unique_customers)
    )

    # Button to trigger validation
    if st.button("Run Validation"):
        results_df = compute_validation(
            model_type=model_type,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            daily_per_customer=daily_per_customer,
            customer_choice=customer_choice
        )

        # Store results in session_state for reuse
        st.session_state["results_df"] = results_df
        st.session_state["model_type"] = model_type
        st.session_state["train_end_date"] = train_end_date
        st.session_state["test_start_date"] = test_start_date
        st.session_state["test_end_date"] = test_end_date
        st.session_state["customer_choice"] = customer_choice

    # If we have results in session_state, display them
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        if results_df.empty:
            st.warning("No forecasts were generated for the selected date range and model.")
            return

        # Compute global metrics
        abs_errors = results_df["error"].abs()
        sq_errors = results_df["error"] ** 2
        mean_ae = np.mean(abs_errors)
        mean_se = np.mean(sq_errors)
        mean_mse = mean_se
        mean_rmse = np.sqrt(mean_mse)

        st.subheader(f"Overall performance for **{st.session_state['model_type']}**:")
        st.write(f"- **Average Absolute Error (AE)**: {mean_ae:.2f}")
        st.write(f"- **Mean Squared Error (MSE)**: {mean_mse:.2f}")
        st.write(f"- **Root MSE (RMSE)**: {mean_rmse:.2f}")

        st.write("### Per-customer results")
        st.dataframe(results_df)

        # Number input for how many customers to show on the bar chart
        st.write("### Error distribution (predicted - actual) per customer")
        max_clients = len(results_df)
        default_value = min(10, max_clients)  # np. 10 lub mniej, gdy klientów jest < 10

        top_n = st.number_input(
            "Number of customers to show on the chart:",
            min_value=1,
            max_value=max_clients,
            value=default_value,
            step=1
        )

        # Sort by absolute error and take top_n
        results_df['abs_error'] = results_df['error'].abs()
        df_sorted = results_df.sort_values('abs_error', ascending=False).head(top_n)

        # Bar chart in Altair
        chart = alt.Chart(df_sorted).mark_bar().encode(
            x=alt.X('error:Q', title='Error (pred - actual)'),
            y=alt.Y('customer_id:N', sort='-x', title='Customer ID'),
            tooltip=['customer_id', 'error', 'actual_3m_sum', 'pred_3m_sum']
        ).properties(
            width=700,
            height=400
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # If a single customer was selected, show daily plot
        if st.session_state["customer_choice"] != all_option:
            df_pred = predict_daily_for_customer(
                customer_id=st.session_state["customer_choice"],
                model_type=st.session_state["model_type"],
                train_end_date=st.session_state["train_end_date"],
                test_start_date=st.session_state["test_start_date"],
                test_end_date=st.session_state["test_end_date"],
                daily_per_customer=daily_per_customer
            )
            if df_pred is not None and not df_pred.empty:
                st.write("### Daily Forecast vs. Actual (selected customer)")
                df_pred_reset = df_pred.reset_index()

                chart_daily = alt.Chart(df_pred_reset).transform_fold(
                    ['actual', 'predicted'],
                    as_=['variable', 'value']
                ).mark_line().encode(
                    x='date:T',
                    y='value:Q',
                    color='variable:N',
                    tooltip=['date:T', 'variable:N', 'value:Q']
                ).interactive()

                st.altair_chart(chart_daily, use_container_width=True)


if __name__ == "__main__":
    main()
