from src.utils.config_loader import load_config
import pandas as pd
from src.training.trainer import train_model
from src.utils.constants import CONFIG_MODEL, CONFIG_TRAIN, TARGET_COLUMN

def load_data(model_name, data_path, train_year, train_month):
    """Take the last 3 months as a validation set

    Args:
        data_path (str): _description_
        train_year (int): Highest year in training data
        train_month (int): Corresponding month

    Returns:
        _type_: _description_
    """
    
    preprocessed_data = pd.read_csv(data_path)
    if model_name == "NeuralNet" :
        valid_year = train_year 
        valid_month = train_month
        
        if 1 <= train_month <=2:
            valid_year = train_year - 1
            valid_month = (train_month-3) % 12 + 1
            train_year = valid_year
            train_month = valid_month-1
        else:
            valid_month -= 2
            train_year = valid_year
            if valid_month!= 0:
                train_month = valid_month-1 
            else :
                train_month = 12
                        
        train = preprocessed_data[
            (preprocessed_data['end_year'] < train_year) |
            ((preprocessed_data['end_year'] == train_year) & (preprocessed_data['end_month'] <= train_month))
        ]
        valid = preprocessed_data[
            (preprocessed_data['start_year'] > valid_year) |
            ((preprocessed_data['start_year'] == valid_year) & (preprocessed_data['start_month'] >= valid_month))
        ]
        X_train = train.drop(columns=[TARGET_COLUMN])  
        y_train = train[TARGET_COLUMN]
        X_valid = valid.drop(columns=[TARGET_COLUMN])
        y_valid = valid[TARGET_COLUMN]
    else :
        X_train = preprocessed_data.drop(columns=[TARGET_COLUMN])  
        y_train = preprocessed_data[TARGET_COLUMN]
        X_valid = pd.DataFrame()
        y_valid = pd.Series()
    
    return X_train.values, y_train.values, X_valid.values, y_valid.values


def main(model_name):
    
    train_config = load_config(CONFIG_TRAIN)
    model_config = load_config(CONFIG_MODEL, model_name)
    
    
    X_train, y_train, X_val, y_val = load_data(
        model_name,
        train_config["data_path"],
        train_config["train_year"],
        train_config["train_month"]
    )
         
    model = train_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_config=model_config
    )
    
    # Save the trained model
    if model_name != "NeuralNet" : # saving is done automatically in neural nets
        model.save_model()

if __name__ == "__main__":
    main()
