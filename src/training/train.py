from src.utils.config_loader import load_config
import pandas as pd
from src.training.trainer import train_model

def load_data(data_path, train_year, train_month):
    """Take the last 3 months as a validation set

    Args:
        data_path (str): _description_
        train_year (int): Highest year in training data
        train_month (int): Corresponding month

    Returns:
        _type_: _description_
    """
    
    preprocessed_data = pd.read_csv(data_path)
    
    test_year = train_year 
    test_month = train_month
    
    if 1 <= train_month <=2:
        test_year = train_year - 1
        test_month = (train_month-3) % 12 + 1
        train_year = test_year
        train_month = test_month-1
    else:
        test_month -= 2
        train_year = test_year
        if test_month!= 0:
            train_month = test_month-1 
        else :
            train_month = 12
                    
    train = preprocessed_data[
        (preprocessed_data['end_year'] < train_year) |
        ((preprocessed_data['end_year'] == train_year) & (preprocessed_data['end_month'] <= train_month))
    ]
    valid = preprocessed_data[
        (preprocessed_data['start_year'] > test_year) |
        ((preprocessed_data['start_year'] == test_year) & (preprocessed_data['start_month'] >= test_month))
    ]
    
    X_train = train.drop(columns=['transactions'])  
    y_train = train['transactions']
    X_valid = valid.drop(columns=['transactions'])
    y_valid = valid['transactions']
    
    return X_train.values, y_train.values, X_valid.values, y_valid.values


def main():
    
    train_config = load_config("configs/train_config.yaml")
    model_config = load_config("configs/model_config.yaml", train_config["model_name"])
    
    
    X_train, y_train, X_val, y_val = load_data(
        train_config["data_path"],
        train_config["train_year"],
        train_config["train_month"]
    )
         
    model = train_model(
        model_name=train_config["model_name"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_config=model_config
    )
    
    # Save the trained model
    model.save_model()

if __name__ == "__main__":
    main()
