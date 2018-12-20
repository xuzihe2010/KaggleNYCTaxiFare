import dataset_reader
import helper_functions
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time

data_raw = dataset_reader.read_train_dataset(5_000_000)
data_test = pd.read_csv("C:/Users/xuzih/Kaggle Data/NYC Taxi Fare/test.csv", parse_dates=["pickup_datetime"])
data_test = helper_functions.feature_engineering(data_test)
data_test = helper_functions.add_time_features(data_test)

data_raw = data_raw[helper_functions.FEATURE_COLS + ['fare_amount']]
X_train, X_valid, y_train, y_valid = train_test_split(data_raw, data_raw.fare_amount,
                                                      test_size=0.2, random_state=1000003)
del data_raw

# Randomized searched params:
start = time.time()
eval_set = [((X_valid[helper_functions.FEATURE_COLS], y_valid))]
xgbr_params = {'subsample': 0.9, 'n_estimators': 10000, 'min_split_loss': 4, 'max_depth': 7, 'max_delta_step': 3,
               'reg_lambda': 3, 'learning_rate': 1.0, 'reg_alpha': 0}
xgbr = xgb.XGBRegressor(random_state=1000003, n_gpus=-1, tree_method="gpu_hist", **xgbr_params)
xgbr.fit(X_train[helper_functions.FEATURE_COLS], y_train)
end = time.time()
print("Train xgboost model takes: ", (end - start) / 60, "mins")

preds = xgbr.predict(data_test[helper_functions.FEATURE_COLS])
sub = pd.DataFrame({'key': list(data_test.key), 'fare_amount': preds})
sub.to_csv('C:/Users/xuzih/Kaggle Data/NYC Taxi Fare/xgboost_pred_sub.csv', index=False)