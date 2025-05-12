import data_loader
import main_eval
from rf import ABRegressorDetector
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

if __name__ == '__main__':
    # Load data
    x_train_full, y_train_full, _ = data_loader.load_train_data("BATADAL", no_transform=True)
    x_test, y_test, _ = data_loader.load_test_data("BATADAL", no_transform=True)

    # Normalize features
    scaler = MinMaxScaler()
    x_train_full = scaler.fit_transform(x_train_full)
    x_test = scaler.transform(x_test)

    # Split benign-validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.3, random_state=42
    )

    best_model = None
    best_val_loss = float('inf')
    best_n_estimators = 10
    max_estimators = 200
    step = 10

    print("(+) Starting AdaBoost training with early stopping...")

    for n_estimators in range(10, max_estimators + 1, step):
        model = ABRegressorDetector(
            n_estimators=n_estimators,
            random_state=42,
            threshold='percentile',  # Use dynamic thresholding
            percentile=95,
            window_length=5         # Optional smoothing
        )
        model.train(x_train, y_train)

        y_val_pred = model.predict(x_val)
        val_loss = ((y_val - y_val_pred) ** 2).mean()
        print(f"    - n_estimators={n_estimators} -> val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_n_estimators = n_estimators
        else:
            print("    (early stopping: validation loss not improving)")
            break

    print(f"(+) Best model found with n_estimators={best_n_estimators}, val_loss={best_val_loss:.6f}")

    # Detect using best model
    anomalies, y_pred, mse = best_model.detect(x_test, y_test)

    # Binary classification-like evaluation
    print("[+] Final F1 Score:", f1_score(y_test, anomalies))
    main_eval.plot_evaluation(y_test, anomalies)

    # Save model and scaler
    joblib.dump(best_model.get_model(), "best_ab_model.joblib")
    joblib.dump(scaler, "scaler.joblib")