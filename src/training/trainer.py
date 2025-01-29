from src.models import RidgeML, RandomForest, XGBoost, NeuralNet

def train_model(model_name, X_train, y_train, X_val, y_val, model_config):
    """
    Train the model based on the selected model type.

    Parameters:
    model_name: str, name of the model (e.g., "xgboost", "ridge").
    X_train: training features.
    y_train: training labels.
    X_val: validation features.
    y_val: validation labels.
    model_config: model-specific configurations (e.g., hyperparameters).

    Returns:
    Trained model.
    """
    
    
    if model_name == "XGBoost":
        model_class = XGBoost
    elif model_name == "RandomForest":
        model_class = RandomForest
    elif model_name == "RidgeML":
        model_class = RidgeML
    elif model_name == "NeuralNet":
        model_class = NeuralNet
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if model_name == "NeuralNet": 
        model = model_class(input_dim=X_train.shape[1], **model_config)
    else:  
        model = model_class(**model_config)
    
    model.logger.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    model.logger.info(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
    model.logger.info(f"Model config: {model_config}")
    
    # Fit the model
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    return model
