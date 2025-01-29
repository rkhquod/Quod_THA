from abc import ABC, abstractmethod
import os
import yaml
import torch
import joblib
import time
import numpy as np
from src.utils.logger import setup_logger 
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MLModel(ABC):
    """
    Abstract base class to define the layout of a machine learning model.
    All machine learning models should inherit from this class and implement the required methods.
    """

    @abstractmethod
    def __init__(self, name: str, config_path="configs/model_config.yaml", **kwargs):
        """
        Initialize the model with necessary parameters.

        Parameters:
        name: Name of the model.
        """
        self.name = name
        self.config_path = config_path
        self.model_save_dir = "artifacts/" + name
        self.scaler_path = os.path.join(self.model_save_dir, "scaler.pkl")
        self.scaler = None
        self.logger = setup_logger(self.name, self.model_save_dir)  # Use the centralized logger setup

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the model on the given data.

        Parameters:
        X: Features (e.g., a DataFrame or array-like structure).
        y: Target labels or values.
        """
        start_time = time.time()
        self.logger.info(f"Training {self.name} model...")
        # Call the subclass's fit implementation
        self._fit_impl(X, y, X_val, y_val)
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds.")
        
    def predict(self, X):
        """
        Predict the target values for the given features.

        Parameters:
        X: Features to predict on (e.g., a DataFrame or array-like structure).

        Returns:
        Predictions (array-like).
        """
        start_time = time.time()
        self.logger.info(f"Making predictions using {self.name} model...")

        # Call the subclass's fit implementation
        predictions = self._predict_impl(X)
        training_time = time.time() - start_time
        self.logger.info(f"Prediction completed in {training_time:.2f} seconds.")
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate the model's performance on the given data.

        Parameters:
        X: Features (e.g., a DataFrame or array-like structure).
        y: True target values.

        Returns:
        Evaluation metric(s).
        """
        start_time = time.time()
        self.logger.info(f"Evaluating {self.name} model...")
        metrics = self._evaluate_impl(X, y)
        
        # Log the evaluation metrics
        if isinstance(metrics, dict):
            # If metrics is a dictionary, log each key-value pair
            self.logger.info("Evaluation metrics:")
            for metric_name, metric_value in metrics.items():
                self.logger.info(f"{metric_name}: {metric_value:.4f}")
        else:
            # If metrics is a single value, log it directly
            self.logger.info(f"Evaluation metric: {metrics:.4f}")
        
        evaluation_time = time.time() - start_time
        self.logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds.") 
        return metrics
    
    
    def save_model(self, save_dir="artifacts"):
        """Save the model, configuration, and scaler to a directory, using only model name."""
        os.makedirs(save_dir, exist_ok=True)

        model_filename = f"{self.name}_model.pkl"
        config_filename = f"{self.name}_config.yaml"
        scaler_filename = "scaler.pkl"
        
        model_path = os.path.join(save_dir, self.name, model_filename)
        config_path = os.path.join(save_dir, self.name, config_filename)
        scaler_path = os.path.join(save_dir, self.name, scaler_filename)

        if hasattr(self.model, 'save'):
            # If the model has a 'save' method (PyTorch models)
            torch.save(self.model.state_dict(), model_path)
        else:
            # Otherwise, save using joblib for scikit-learn models
            joblib.dump(self.model, model_path)
        
        # Save the configuration (hyperparameters)
        with open(config_path, "w") as file:
            yaml.safe_dump({self.name: self.params}, file)

        # Save the scaler if it exists
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load_model(cls, save_dir="model_artifacts"):
        """Load the model, configuration, and scaler (if applicable) from a directory."""
        
        model_name = cls.__name__  # Use class name if not provided
        # Generate the paths based on model_name
        model_path = os.path.join(save_dir, model_name, f"{model_name}_model.pkl")
        config_path = os.path.join(save_dir, model_name, f"{model_name}_config.yaml")

        # Load the configuration (hyperparameters)
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)[model_name]

        # Initialize the model
        model = cls(**config)
        # Load the model (scikit-learn models use joblib)
        model.model = joblib.load(model_path)

        print(f"Model and configuration loaded from {save_dir}")
        return model
    
    def get_params(self):
        return self.params
    
    def _fit_impl(self, X, y, X_val=None, y_val=None):
        """
        Subclass-specific implementation of the fit method.
        """
        self.model = self.model.fit(X, y)

    def _predict_impl(self, X):
        """
        Subclass-specific implementation of the predict method.
        """
        predictions = self.model.predict(X)
        predictions = np.clip(predictions, 0, None)
        return predictions

    def _evaluate_impl(self, X, y):
        """
        Subclass-specific implementation of the evaluate method.
        """
        predictions = self.predict(X)
        mae =  mean_absolute_error(y, predictions) 
        rmse =  mean_squared_error(y, predictions) 
        return{"MAE" : mae, "MSE" : rmse}