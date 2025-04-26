import data_loader
import main_eval
from rf import RandomForestModel  # đảm bảo file rf_model.py chứa class đã tạo trước đó
import numpy as np

if __name__ == '__main__':
    """
    Load data and preprocessing
    x_train, y_train: fully labeled data for training the model (0 if normal else 1)
    x_test: used for prediction
    y_test: ground truth (0 if normal else 1)
    y_pred: model predictions (0 if normal else 1)
    """

    # Load training and test data (FULLY labeled)
    x_train, y_train, _ = data_loader.load_train_data("BATADAL", no_transform=False)
    x_test, y_test, _ = data_loader.load_test_data("BATADAL", no_transform=False)

    # Create model
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42,
        'verbose': 0
    }
    rf_model = RandomForestModel(**params)

    # Train model
    rf_model.train(x_train, y_train)

    # Predict
    y_pred = rf_model.predict(x_test)

    # Evaluate
    main_eval.plot_evaluation(y_test, y_pred)
