from abc import ABC, abstractmethod
from functools import wraps
import os
import yaml
import torch
import joblib

class MLModel(ABC):
    """
    Abstract base class to define the layout of a machine learning model.
    All machine learning models should inherit from this class and implement the required methods.
    """

    @abstractmethod
    def __init__(self, name: str, config_path="models/config.yaml", **kwargs):
        """
        Initialize the model with necessary parameters.

        Parameters:
        name: Name of the model.
        """
        self.name = name
        self.config_path = config_path
        self.model_save_dir = name + "_Model"
        
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the model on the given data.

        Parameters:
        X: Features (e.g., a DataFrame or array-like structure).
        y: Target labels or values.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predict the target values for the given features.

        Parameters:
        X: Features to predict on (e.g., a DataFrame or array-like structure).

        Returns:
        Predictions (array-like).
        """
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model's performance on the given data.

        Parameters:
        X: Features (e.g., a DataFrame or array-like structure).
        y: True target values.

        Returns:
        Evaluation metric(s).
        """
        pass
    
    def save_model(self, save_dir="model_artifacts"):
        """Save the model and configuration to a directory, using only model name."""
        os.makedirs(save_dir, exist_ok=True)

        model_filename = f"{self.name}_model.pkl"
        config_filename = f"{self.name}_config.yaml"
        model_path = os.path.join(save_dir, model_filename)
        config_path = os.path.join(save_dir, config_filename)

        if hasattr(self.model, 'save'):
            # If the model has a 'save' method (PyTorch models)
            torch.save(self.model.state_dict(), model_path)
        else:
            # Otherwise, save using joblib for scikit-learn models
            joblib.dump(self.model, model_path)
        
        # Save the configuration (hyperparameters)
        with open(config_path, "w") as file:
            yaml.safe_dump({self.name: self.params}, file)

    @classmethod
    def load_model(cls, save_dir="model_artifacts", model_name=None):
        """Load the model and configuration from a directory."""
        if model_name is None:
            model_name = cls.__name__  # Use class name if not provided

        # Generate the paths based on model_name
        model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
        config_path = os.path.join(save_dir, f"{model_name}_config.yaml")

        # Load the configuration (hyperparameters)
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)[model_name]
        
        # Initialize the model (scikit-learn models and others)
        model = cls(**config)
        # Load the model (scikit-learn models use joblib)
        model.model = joblib.load(model_path)
        
        print(f"Model and configuration loaded from {save_dir}")
        return model