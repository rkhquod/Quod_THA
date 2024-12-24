"""
1) Prepare the data for modeling
2) Train MULTIPLE models (e.g., Linear Regression & Random Forest)
3) Compare their performance
4) Serialize (save) each model as a pickle file
"""

import math
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_data_for_modeling(df: pd.DataFrame):
    """
    Prepare data for a 'next 3 months' transactions prediction.

    Steps:
     - Aggregate data by customer + monthly period.
     - Create a pivot with months as columns, counts as values.
     - Use all but the last 3 months as features (X)
       and the sum of the last 3 months as the target (Y).
    """
    # Convert the date to monthly period 
    df['month_period'] = df['date'].dt.to_period('M')

    # Group by customer_id and monthly period
    monthly_data = (
        df.groupby(['customer_id', 'month_period'])
          .size()
          .reset_index(name='transaction_count')
    )

    # Convert period to string for pivot
    monthly_data['month_period'] = monthly_data['month_period'].astype(str)

    # Create pivot table
    pivot_df = monthly_data.pivot(
        index='customer_id',
        columns='month_period',
        values='transaction_count'
    ).fillna(0)

    if pivot_df.shape[1] > 3:
        X = pivot_df.iloc[:, :-3]
        Y = pivot_df.iloc[:, -3:].sum(axis=1)
    else:
        X = pivot_df
        Y = pivot_df.sum(axis=1)

    return X, Y


def train_and_compare_models(df: pd.DataFrame):
    """
    Train at least two models (LinearRegression, RandomForestRegressor),
    compare their performance, and return them as a dict:

    Returns:
    --------
    models_dict: dict
        {
          'linear_regression': trained_model,
          'random_forest': trained_model
        }
    metrics_dict: dict
        {
          'linear_regression': (MAE, RMSE),
          'random_forest': (MAE, RMSE)
        }
    """

    X, Y = prepare_data_for_modeling(df)

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Define multiple models
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    models_dict = {}
    metrics_dict = {}

    print("=== Training Multiple Models ===")

    for model_name, model_obj in models.items():
        # Train
        model_obj.fit(X_train, Y_train)

        # Predict
        Y_pred = model_obj.predict(X_test)

        # Evaluate
        mae = mean_absolute_error(Y_test, Y_pred)
        rmse = math.sqrt(mean_squared_error(Y_test, Y_pred))

        # Store model & metrics
        models_dict[model_name] = model_obj
        metrics_dict[model_name] = (mae, rmse)

        print(f"Model: {model_name}")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print("--------------")

    return models_dict, metrics_dict


def save_models(models_dict: dict):
    """
    Saves each model in models_dict as a pickle file.
    The key of the dictionary is used as the filename prefix.
    """
    for model_name, model_obj in models_dict.items():
        filename = f"{model_name}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model_obj, f)
        print(f"Saved '{model_name}' model to '{filename}'")

