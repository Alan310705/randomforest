import data_loader
import main_eval
from rf import RFRegressorDetector  # this should use RandomForestRegressor
from sklearn.metrics import f1_score
import numpy as np

if __name__ == '__main__':
    # Load training and test data
    x_train, y_train, _ = data_loader.load_train_data("BATADAL", no_transform=False)
    x_test, y_test, _ = data_loader.load_test_data("BATADAL", no_transform=False)

    # Create regression model
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42
    }
    rf_model = RFRegressorDetector(**params)

    # Train the model
    rf_model.train(x_train, y_train)

    # Predict (continuous values)
    y_pred_cont = rf_model.predict(x_test)

    # Thresholding: Convert to binary classification for F1-score
    threshold = 0.65  # or choose based on ROC/validation
    y_pred_binary = (y_pred_cont >= threshold).astype(int)

    # Evaluate
    print("F1 Score:", f1_score(y_test, y_pred_binary))
    main_eval.plot_evaluation(y_test, y_pred_binary)
