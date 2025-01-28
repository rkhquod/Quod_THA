from models import MLModel
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import yaml

class RandomForestModel(MLModel):
    def __init__(self, **kwargs):
        super().__init__(name="RandomForest")
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        default_params = config.get('RandomForest', {})
        self.params = {**default_params, **kwargs}
        self.model = RandomForestRegressor(**self.params)
        
    def get_params(self):
        return self.params
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return mean_absolute_error(y, predictions) 

class XGBoostModel(MLModel):
    def __init__(self, **kwargs):
        super().__init__(name="XGBoost")
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        default_params = config.get('XGBoost', {})
        self.params = {**default_params, **kwargs}
        self.model = XGBRegressor(**self.params)
        
    def get_params(self):
        return self.params
    
    def fit(self, X, y, X_val=None, y_val=None):
        self.model = self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mae =  mean_absolute_error(y, predictions) 
        return mae
