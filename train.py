from models import RidgeModel, RandomForestModel, XGBoostModel, NeuralNetModel
import argparse
import pandas as pd



def train_model(X_train, y_train, X_val, y_val, **kwargs):
    """
    Train a machine learning model.

    Parameters:
    model_class: The class of the model to train (e.g., EnsembleModel, NeuralNet).
    train_data: Tuple of (X_train, y_train) for training.
    val_data: Optional tuple of (X_val, y_val) for validation.
    **kwargs: Additional arguments to pass to the model's constructor.
    """
    model_name = kwargs["model_name"]
    if model_name == "xgboost":
        model_class = XGBoostModel
    elif model_name == "random_forest":
        model_class = RandomForestModel
    elif model_name == "ridge":
        model_class = RidgeModel
    elif model_name == "neural_net":
        model_class = NeuralNetModel
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_name == "neural_net":
        input_dim = X_train.shape[1]
        model = model_class(input_dim=input_dim, **kwargs)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else :
        model = model_class(**kwargs)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    return model

def load_data(data_path, train_year, train_month):
    
    preprocessed_data = pd.read_csv(data_path)
    
    train = preprocessed_data[
    (preprocessed_data['end_year'] < train_year) |
    ((preprocessed_data['end_year'] == train_year) & (preprocessed_data['end_month'] <= train_month))
]
    
    valid = preprocessed_data[
        (preprocessed_data['start_year'] > train_year) |
        ((preprocessed_data['start_year'] == train_year) & (preprocessed_data['start_month'] > train_month))
    ]
    
    X_train = train.drop(columns=['transactions'])  
    y_train = train['transactions']
    X_valid = valid.drop(columns=['transactions'])
    y_valid = valid['transactions']
    
    return X_train.values, y_train.values, X_valid.values, y_valid.values


def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument("--model_name", type=str,
                        choices=["xgboost", "random_forest","ridge","neural_net"],
                        help="Type of model to train.", default="xgboost")
    parser.add_argument("--train_year", type=int, help="Last year for training transaction data", default=2018)
    parser.add_argument("--train_month", type=int, help="Last month for training transaction data", default=10)
    parser.add_argument("--data_path", type=str, help="Path containing preprocessed data", default="preprocessed_train.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    X_train, y_train, X_val, y_val = load_data(args.data_path, args.train_year, args.train_month)
    print("Train data shape : ", X_train.shape)
    print("Validation data shape : ", X_val.shape)
    model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        **vars(args)
    )
    model.save_model(model.model_save_dir)

if __name__ == "__main__":
    main()
