from src.models import MLModel
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from src.utils.config_loader import load_config

class RidgeML(MLModel):
    def __init__(self, **kwargs):
        super().__init__(name="RidgeML")
        config = load_config(self.config_path)
        default_params = config.get('Ridge', {})
        self.params = {**default_params, **kwargs}
        self.model = Ridge(**self.params)
    