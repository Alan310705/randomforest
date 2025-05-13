import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RFRegressorDetector:
    """
    Random Forest Regressor for ICS anomaly detection.
    Predicts the next value of the first sensor channel.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        print("(+) Initializing Random Forest Regressor...")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.threshold = None
        self.window = 1

    def train(self, X_train):
        """
        Train the model to predict the next timestep value of the first sensor.
        """
        print("(+) Training Random Forest Regressor...")
        X = X_train[:-1, :]
        y = X_train[1:, 0]  # predict next step of sensor 0
        self.model.fit(X, y)

    def detect(self, X_test, X_val=None, quantile=0.95, window=1):
        """
        Predict anomalies by comparing squared error to dynamic threshold.
        """
        if X_val is not None:
            Xv = X_val[:-1, :]
            yv = X_val[1:, 0]
            preds_val = self.model.predict(Xv)
            val_errors = (preds_val - yv) ** 2
            self.threshold = np.quantile(val_errors, quantile)

        # Predict test data
        X = X_test[:-1, :]
        y_true = X_test[1:, 0]
        preds = self.model.predict(X)
        errors = (preds - y_true) ** 2

        raw_flags = (errors > self.threshold).astype(int)
        raw_flags = np.concatenate([[0], raw_flags])  # Align with input length

        # Apply sliding window smoothing
        if window > 1:
            flags = np.zeros_like(raw_flags)
            for i in range(len(raw_flags)):
                start = max(0, i - window + 1)
                flags[i] = raw_flags[start:i + 1].max()
        else:
            flags = raw_flags

        return flags, preds, errors

    def get_model(self):
        return self.model


if __name__ == "__main__":
    print("Not a standalone script.")
