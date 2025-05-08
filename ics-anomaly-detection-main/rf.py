# numpy stack
import numpy as np

# Ignore ugly futurewarnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# sklearn for Random Forest
from sklearn.ensemble import RandomForestRegressor

# Random Forest Regressor class
class RFRegressorDetector(object):
    """ sklearn-based Random Forest Regressor for ICS anomaly detection.

        Attributes:
        params: dictionary with parameters defining the model structure,
    """
    def __init__(self, **kwargs):
        """ Constructor: stores parameters and initializes RF model. """
        print("(+) Initializing Random Forest Regressor...")

        # Default parameter values
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'threshold': 0.05,  # Threshold for MSE
            'verbose': 0
        }

        # Update parameters with kwargs
        for key, item in kwargs.items():
            params[key] = item
        self.params = params

    def create_model(self):
        """ Creates RF Regressor model using sklearn. """
        print("(+) Creating Random Forest Regressor model...")
        self.rf = RandomForestRegressor(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            random_state=self.params['random_state']
        )
        return self.rf

    def train(self, x_train, y_train):
        """ Train the RF model on labeled data. """
        self.create_model()
        print("(+) Training RF model...")
        self.rf.fit(x_train, y_train)

    def predict(self, x_test):
        """ Predict numeric output using trained RF model. """
        return self.rf.predict(x_test)

    def detect(self, x_test, y_true):
        """ Perform anomaly detection by comparing MSE with threshold. """
        y_pred = self.predict(x_test)
        mse = (y_pred - y_true) ** 2
        anomalies = (mse > self.params['threshold']).astype(int)
        return anomalies, y_pred, mse

    def get_rf(self):
        """ Return the internal Random Forest model. """
        return self.rf

if __name__ == "__main__":
    print("Not a main file.")