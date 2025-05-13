import numpy as np
from sklearn.ensemble import AdaBoostRegressor

class AdaBoostRegressorDetector:
    """
    AdaBoost-based Detector for ICS anomaly detection using one-step-ahead regression.
    Predicts the next value of the first sensor channel.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        print("(+) Initializing AdaBoost Regressor Detector...")
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
        self.model = None
        self.threshold = None
        self.window = 1

    def create_model(self):
        """
        Creates the AdaBoost Regressor model.
        """
        print("(+) Creating AdaBoost Regressor model...")
        self.model = AdaBoostRegressor(
            n_estimators=self.params['n_estimators'],
            learning_rate=self.params['learning_rate'],
            random_state=self.params['random_state']
        )

    def train(self, X_train):
        """
        Train the AdaBoost model to predict the next timestep value of the first sensor.
        """
        if self.model is None:
            self.create_model()
        print("(+) Training AdaBoost Regressor...")
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        self.model.fit(X, y)

    def detect(self, X_test, X_val=None, quantile=0.95, window=1):
        """
        Detect anomalies in the test set using prediction error thresholding and sliding window.
        """
        self.window = window

        # Estimate threshold from validation set
        if X_val is not None:
            Xv = X_val[:-1, :]
            yv = X_val[1:, 0]
            preds_val = self.model.predict(Xv)
            val_errors = (preds_val - yv) ** 2
            self.threshold = np.quantile(val_errors, quantile)
            print(f"(i) Threshold (quantile={quantile}): {self.threshold:.5f}")

        # Predict on test set
        X = X_test[:-1, :]
        y_true = X_test[1:, 0]
        preds = self.model.predict(X)
        errors = (preds - y_true) ** 2

        # Raw flags: error > threshold
        raw_flags = (errors > self.threshold).astype(int)
        raw_flags = np.concatenate([[0], raw_flags])  # For alignment with test set

        # Apply sliding window smoothing
        if self.window > 1:
            flags = np.zeros_like(raw_flags)
            for i in range(len(raw_flags)):
                start = max(0, i - self.window + 1)
                flags[i] = raw_flags[start:i + 1].max()
        else:
            flags = raw_flags

        return flags, preds, errors

    def get_model(self):
        return self.model


if __name__ == "__main__":
    print("Not a standalone script.")
