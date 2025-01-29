from src.models import MLModel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.utils.config_loader import load_config


class RandomForest(MLModel):
    def __init__(self, **kwargs):
        super().__init__(name="RandomForest")
        config = load_config(self.config_path)
        default_params = config.get('RandomForest', {})
        self.params = {**default_params, **kwargs}
        self.model = RandomForestRegressor(**self.params)
        

class XGBoost(MLModel):
    def __init__(self, **kwargs):
        super().__init__(name="XGBoost")
        config = load_config(self.config_path)
        default_params = config.get('XGBoost', {})
        self.params = {**default_params, **kwargs}
        self.model = XGBRegressor(**self.params)
        

