import argparse
import pandas as pd
from models import RidgeModel, RandomForestModel, XGBoostModel, NeuralNetModel
import os
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

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained machine learning model.")
    parser.add_argument("--model_name", type=str, 
                        choices=["xgboost", "random_forest", "ridge", "neural_net"], 
                        help="Name of the model to evaluate.",
                        default="xgboost")
    parser.add_argument("--test_data_path", type=str, 
                        help="Path to the preprocessed test data file.", 
                        default="preprocessed_test.csv")
    parser.add_argument("--model_dir", default="model_artifacts", 
                        help="Directory where the model artifacts are saved.")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load the preprocessed test data
    X_test, y_test = load_test_data(args.test_data_path)
    X_test, y_test = X_test.values, y_test.values
    
    
    # Load the appropriate model class
    if args.model_name == "xgboost":
        model_class = XGBoostModel()
    elif args.model_name == "random_forest":
        model_class = RandomForestModel()
    elif args.model_name == "ridge":
        model_class = RidgeModel()
    elif args.model_name == "neural_net":
        model_class = NeuralNetModel(input_dim=X_test.shape[1])
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    # Load the trained model using the class method
    model = model_class.load_model(save_dir=model_class.model_save_dir, model_name=model_class.name)
    scaler_path = os.path.join(model_class.model_save_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)
    if args.model_name == "neural_net":
        X_test = scaler.transform(X_test)
        
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print(f"Test MAE for {args.model_name}: {metrics}")

if __name__ == "__main__":
    main()