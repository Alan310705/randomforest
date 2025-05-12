import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.ndimage import uniform_filter1d

class ABRegressorDetector(object):
    """ sklearn-based AdaBoost Regressor for ICS anomaly detection with extended functionality. """

    def __init__(self, **kwargs):
        print("(+) Initializing AdaBoost Regressor...")

        # Default parameters
        params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'loss': 'linear',
            'random_state': 42,
            'threshold': 0.05,  # Threshold for MSE-based anomaly detection
            'window_size': 1,   # Default no smoothing
            'percentile': None  # If None, use MSE threshold; else use percentile-based thresholding
        }

        for key, item in kwargs.items():
            params[key] = item
        self.params = params

    def create_model(self):
        print("(+) Creating AdaBoost Regressor model...")
        base_estimator = DecisionTreeRegressor(max_depth=4)
        self.model = AdaBoostRegressor(
            base_estimator=base_estimator,
            n_estimators=self.params['n_estimators'],
            learning_rate=self.params['learning_rate'],
            loss=self.params['loss'],
            random_state=self.params['random_state']
        )
        return self.model

    def train(self, x_train, y_train):
        self.create_model()
        print("(+) Training AdaBoost model...")
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def detect(self, x_test, y_true):
        y_pred = self.predict(x_test)
        mse = (y_pred - y_true) ** 2

        # Optional window smoothing
        if self.params['window_size'] > 1:
            mse = self.apply_window_smoothing(mse, self.params['window_size'])

        # Thresholding
        if self.params['percentile'] is not None:
            anomalies, threshold = self.threshold_by_percentile(mse, self.params['percentile'])
        else:
            threshold = self.params['threshold']
            anomalies = (mse > threshold).astype(int)

        print(f"(+) Detection threshold: {threshold:.4f}")
        return anomalies, y_pred, mse

    def apply_window_smoothing(self, scores, window_size):
        """ Smooth score sequence to reduce false positives. """
        return uniform_filter1d(scores, size=window_size)

    def threshold_by_percentile(self, scores, percentile):
        """ Compute binary predictions using a percentile-based threshold. """
        threshold = np.percentile(scores, percentile)
        return (scores >= threshold).astype(int), threshold

    def tune_hyperparameters(self, x_train, y_train):
        print("(+) Tuning hyperparameters with GridSearchCV...")

        base_estimator = DecisionTreeRegressor(max_depth=4)
        model = AdaBoostRegressor(base_estimator=base_estimator, random_state=self.params['random_state'])

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1.0],
            'loss': ['linear', 'square', 'exponential']
        }

        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=mse_scorer,
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(x_train, y_train)
        print("(+) Best Parameters:", grid_search.best_params_)
        print("    Best CV MSE:", -grid_search.best_score_)

        self.model = grid_search.best_estimator_
        return self.model

    def get_model(self):
        return self.model

if __name__ == "__main__":
    print("Not a main file.")
