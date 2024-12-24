"""
Loads any saved model, performs predictions, and evaluates its performance.
"""

import pickle
import math

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_train import prepare_data_for_modeling

def evaluate_model(df: pd.DataFrame, model_path: str):
    """
    Load a saved model (model_path), evaluate on the entire DataFrame (or a subset),
    and plot predictions vs. actual.
    """
    X, Y = prepare_data_for_modeling(df)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict
    Y_pred = model.predict(X)

    # Evaluate
    mae = mean_absolute_error(Y, Y_pred)
    rmse = math.sqrt(mean_squared_error(Y, Y_pred))

    print(f"=== Evaluating {model_path} ===")
    print(f"Data samples: {X.shape[0]}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}\n")

    # Scatter plot of Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(Y, Y_pred, alpha=0.5, color='purple')
    plt.title(f"Predictions vs. Actual - {model_path}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
