import pickle
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def load_train_data(dataset_name, scaler=None, train_shuffle=True, no_transform=False, verbose=False):

    if verbose:
        print('Loading {} train data...'.format(dataset_name))

    if scaler is None:
        if verbose:
            print("No scaler provided. Using default sklearn.preprocessing.StandardScaler")
        scaler = StandardScaler()

    if dataset_name == 'BATADAL':
        try:
            df_train = pd.read_csv("ics-anomaly-detection-main/" + dataset_name + "/test_dataset_2.csv", 
                                   parse_dates=['DATETIME'], dayfirst=True)
        except FileNotFoundError:
            raise SystemExit("Unable to find BATADAL train dataset. Did you unpack BATADAL.tar.gz?")
        
        sensor_cols = [col for col in df_train.columns if col not in ['Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
        target_col = 'ATT_FLAG'

    elif dataset_name == 'SWAT':
        try:
            df_train = pd.read_csv("" + dataset_name + "/SWATv0_train.csv", dayfirst=True)
        except FileNotFoundError:
            raise SystemExit("Unable to find SWAT train dataset.")
        sensor_cols = [col for col in df_train.columns if col not in ['Timestamp', 'Normal/Attack']]
        target_col = 'Normal/Attack'

    elif dataset_name == 'WADI':
        try:
            df_train = pd.read_csv("ics-anomaly-detection-main/" + dataset_name + "/WADI_train.csv")
        except FileNotFoundError:
            raise SystemExit("Unable to find WADI train dataset.")
        remove_list = ['Row', 'Date', 'Time', 'Attack', '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
        sensor_cols = [col for col in df_train.columns if col not in remove_list]
        target_col = 'Attack'

    else:
        raise SystemExit(f"Cannot find dataset name: {dataset_name}.")

    # scale sensor data
    if no_transform:
        X = pd.DataFrame(index=df_train.index, columns=sensor_cols, data=df_train[sensor_cols].values)
    else:
        X_prescaled = df_train[sensor_cols].values
        X = pd.DataFrame(index=df_train.index, columns=sensor_cols, data=scaler.fit_transform(X_prescaled))
        # save scaler
        pickle.dump(scaler, open(f'ics-anomaly-detection-main/models/{dataset_name}_scaler.pkl', 'wb'))
        if verbose:
            print('Saved scaler to {}.'.format(f'ics-anomaly-detection-main/models/{dataset_name}_scaler.pkl'))

    # For regression: use original flag as target (can be count or continuous if modified)
    y = df_train[target_col].values.astype(float)

    return X.values, y, sensor_cols

def load_test_data(dataset_name, scaler=None, no_transform=False, verbose=False):
    if verbose:
        print('Loading {} test data...'.format(dataset_name))

    if scaler is None:
        if verbose:
            print('No scaler provided, loading from saved model...')
        try:
            scaler = pickle.load(open(f'ics-anomaly-detection-main/models/{dataset_name}_scaler.pkl', "rb"))
        except FileNotFoundError:
            raise SystemExit(f"Scaler for {dataset_name} not found. Train first to generate scaler.")

    if dataset_name == 'BATADAL':
        try:
            df_test = pd.read_csv("ics-anomaly-detection-main/" + dataset_name + "/test_dataset_1.csv", 
                                  parse_dates=['DATETIME'], dayfirst=True)
        except FileNotFoundError:
            raise SystemExit("Cannot find BATADAL test dataset.")
        sensor_cols = [col for col in df_test.columns if col not in ['Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
        target_col = 'ATT_FLAG'

    elif dataset_name == 'SWAT':
        try:
            df_test = pd.read_csv("" + dataset_name + "/SWATv0_test.csv")
        except FileNotFoundError:
            raise SystemExit("Cannot find SWAT test dataset.")
        sensor_cols = [col for col in df_test.columns if col not in ['Timestamp', 'Normal/Attack']]
        target_col = 'Normal/Attack'

    elif dataset_name == 'WADI':
        try:
            df_test = pd.read_csv("ics-anomaly-detection-main/" + dataset_name + "/WADI_test.csv")
        except FileNotFoundError:
            raise SystemExit("Cannot find WADI test dataset.")
        remove_list = ['Row', 'Date', 'Time', 'Attack', '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
        sensor_cols = [col for col in df_test.columns if col not in remove_list]
        target_col = 'Attack'

    else:
        raise SystemExit(f"Unknown dataset: {dataset_name}")

    # scale test data
    if no_transform:
        X_test = df_test[sensor_cols].values
    else:
        X_test = scaler.transform(df_test[sensor_cols].values)

    y_test = df_test[target_col].values.astype(float)

    return X_test, y_test, sensor_cols
