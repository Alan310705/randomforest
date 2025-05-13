import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from itertools import product
from tqdm import tqdm

class AdaBoostRegressorDetector:
    """
    AdaBoost-based Detector for ICS anomaly detection using one-step-ahead regression.
    Predicts the next value of the first sensor channel.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        print("(+) Initializing AdaBoost Regressor model...")
        self.model = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.threshold = None
        self.window_length = 1

    def train(self, X_train):
        """
        Train the AdaBoost model to predict the next timestep of the first sensor feature.
        """
        print("(+) Training AdaBoost Regressor...")
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        self.model.fit(X, y)

    def detect(self, X_test, X_val=None, quantile=0.95, window=1):
        """
        Detect anomalies in the test set using prediction error thresholding and sliding window.
        """
        if X_val is not None:
            Xv = X_val[:-1, :]
            yv = X_val[1:, 0]
            preds = self.model.predict(Xv)
            errors = (preds - yv) ** 2
            self.threshold = np.quantile(errors, quantile)

        # Test prediction
        X = X_test[:-1, :]
        y_true = X_test[1:, 0]
        preds = self.model.predict(X)
        errors = (preds - y_true) ** 2

        raw_flags = (errors > self.threshold).astype(int)
        raw_flags = np.concatenate([[0], raw_flags])  # prepend 0 for alignment

        # Apply sliding window
        if window > 1:
            flags = np.zeros_like(raw_flags)
            for i in range(len(raw_flags)):
                start = max(0, i - window + 1)
                flags[i] = raw_flags[start:i+1].max()
        else:
            flags = raw_flags

        return flags, preds, errors

    def hyperparameter_tuning(self, X_train, X_val, patience=3):
        """
        Grid search for best (n_estimators, learning_rate) using validation MSE.
        """
        print("(+) Tuning AdaBoost hyperparameters...")
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        Xv = X_val[:-1, :]
        yv = X_val[1:, 0]

        best_model = None
        best_mse = float('inf')
        best_params = {}
        no_improve_count = 0

        n_estimators_list = [50, 100, 150, 200]
        learning_rates = [0.01, 0.05, 0.1, 0.2]

        for n, lr in tqdm(product(n_estimators_list, learning_rates), total=len(n_estimators_list) * len(learning_rates), desc="Hyperparameter tuning"):
            model = AdaBoostRegressor(n_estimators=n, learning_rate=lr, random_state=42)
            model.fit(X, y)
            preds = model.predict(Xv)
            mse = mean_squared_error(yv, preds)
            print(f"-> n_estimators={n}, learning_rate={lr}, Val MSE={mse:.5f}")
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = {'n_estimators': n, 'learning_rate': lr}
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

        print(f"Best AdaBoost model: n_estimators={best_params['n_estimators']}, learning_rate={best_params['learning_rate']}")
        self.model = best_model

    def get_model(self):
        return self.model
