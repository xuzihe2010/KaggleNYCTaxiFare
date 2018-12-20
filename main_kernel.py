from sklearn.ensemble import GradientBoostingRegressor
import dataset_reader
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import helper_functions
import time
import functools

data_raw = dataset_reader.read_train_dataset(110_000)
data_test = pd.read_csv("C:/Users/xuzih/Kaggle Data/NYC Taxi Fare/test.csv", parse_dates=["pickup_datetime"])
data_test = helper_functions.feature_engineering(data_test)
data_test = helper_functions.add_time_features(data_test)

data_raw = data_raw[helper_functions.FEATURE_COLS + ['fare_amount']]
print(data_raw.shape)
print(data_raw.describe())

# params_grid = {
#     'loss': ['ls', 'lad', 'huber', 'quantile'],
#     'learning_rate': [0.2 * x for x in range(1, 6)],
#     'n_estimators': list(range(100, 1000, 100)) + list(range(1000, 2200, 200)),
#     'subsample': [0.1 * x for x in range(1, 10)],
#     'min_samples_split': range(2, 10, 2),
#     'min_samples_leaf': range(1, 20, 2),
#     'max_depth': range(1, 10, 2),
#     'max_features': ['auto', 'sqrt', None] + [0.1 * x for x in range(1, 10)]
# }
#
# # Define sklearn GradientBoosting regressor and Randomized search cv
# gbr = GradientBoostingRegressor(random_state=1000003, n_iter_no_change=5)
# rscv = RandomizedSearchCV(estimator=gbr, param_distributions=params_grid,
#                           n_iter=100, n_jobs=-1, scoring="neg_mean_squared_error",
#                           verbose=1, random_state=1000003)

params_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.5, 1],
    'subsample': [0.5, 1],
    'gamma': [0, 0.5],    # PTAL
    'min_child_weight': [1, 3],
    'colsample_bytree': [0.5, 1],
    'reg_lambda': [1, 1.5],
    'reg_alpha': [0, 0.5],
}

total = functools.reduce(lambda x, y : x * y, [len(num) for name, num in params_grid.items()])
print("Total number of params:", total)

start = time.time()
# Define sklearn GradientBoosting regressor and Randomized search cv
xgbr = xgb.XGBRegressor(early_stopping_rounds=10, eval_metric="rmse",  random_state=1000003,  n_jobs=-1)
rscv = RandomizedSearchCV(estimator=xgbr, param_distributions=params_grid,
                          n_iter=300, n_jobs=-1, scoring="neg_mean_squared_error",
                          verbose=1, random_state=1000003)
end = time.time()
print('start training: ', end - start)

# Run hyperparameter seletion.
start = time.time()
rscv.fit(data_raw[helper_functions.FEATURE_COLS], data_raw.fare_amount)
end = time.time()

print("Hyperparameter selection time in minutes: ", (end - start)/60)
print(rscv.best_estimator_)
print(rscv.best_score_)
print(rscv.best_params_)