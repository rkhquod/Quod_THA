from . import MLModel
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import yaml

class RidgeModel(MLModel):
    def __init__(self, **kwargs):
        super().__init__(name="Ridge")
        with open('models/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        default_params = config.get('RandomForest', {})
        self.params = {**default_params, **kwargs}
        self.model = Ridge(self.params)
    
    def get_params(self):
        return self.params
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return mean_absolute_error(y, predictions) 
    
