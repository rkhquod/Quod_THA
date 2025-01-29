import pandas as pd
from src.models import RidgeML, RandomForest, XGBoost, NeuralNet
from src.utils.config_loader import load_config
import joblib

def load_test_data(data_path):
    """
    Load the preprocessed test data.

    Parameters:
    data_path: Path to the preprocessed test data file.

    Returns:
    X_test: Features for testing.
    y_test: True target values for testing.
    """
    test_data = pd.read_csv(data_path)

    # Assuming the target column is named 'transactions'
    X_test = test_data.drop(columns=['transactions'])
    y_test = test_data['transactions']

    return X_test, y_test

def main():
    
    # Load the evaluation config
    eval_config = load_config("configs/evaluation_config.yaml")

    # Load the preprocessed test data
    X_test, y_test = load_test_data(eval_config["test_data_path"])
    X_test, y_test = X_test.values, y_test.values

    # Load the appropriate model class
    if eval_config["model_name"] == "XGBoost":
        model_class = XGBoost()
    elif eval_config["model_name"] == "RandomForest":
        model_class = RandomForest()
    elif eval_config["model_name"] == "RidgeML":
        model_class = RidgeML()
    elif eval_config["model_name"] == "NeuralNet":
        model_class = NeuralNet(input_dim=X_test.shape[1])
    else:
        raise ValueError(f"Unknown model name: {eval_config['model_name']}")

    # Load the trained model
    model = model_class.load_model(save_dir=eval_config["model_dir"])

    if eval_config["model_name"] == "NeuralNet":
        scaler = joblib.load(model.scaler_path)
        X_test = scaler.transform(X_test)
    metrics = model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()