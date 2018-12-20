import pandas as pd
import dask.dataframe as dd
import helper_functions
import time
import os

TRAIN_DATASET_PATH = "C:/Users/xuzih/Kaggle Data/NYC Taxi Fare/train.csv"
print(os.listdir("C:/Users/xuzih/Kaggle Data/NYC Taxi Fare/"))

def read_train_dataset(nrows = None):
    traintypes = {'fare_amount': 'float32',
                  'pickup_datetime': 'str',
                  'pickup_longitude': 'float32',
                  'pickup_latitude': 'float32',
                  'dropoff_longitude': 'float32',
                  'dropoff_latitude': 'float32',
                  'passenger_count': 'uint8'}
    cols = list(traintypes.keys())

    time1 = time.time()
    if nrows is not None:
        ddf_pd = dd.read_csv(TRAIN_DATASET_PATH, usecols=cols, dtype=traintypes).head(n=nrows)
    else:
        ddf_pd = dd.read_csv(TRAIN_DATASET_PATH, usecols=cols, dtype=traintypes).compute()
    ddf_pd.index = pd.RangeIndex(start=0, stop=len(ddf_pd))
    ddf_pd['pickup_datetime'] = ddf_pd['pickup_datetime'].str.slice(0, 16)
    ddf_pd['pickup_datetime'] = pd.to_datetime(ddf_pd['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    time2 = time.time()
    print("Read dataframe with Dask time: ", (time2 - time1) / 60, "mins")
    ddf_pd = helper_functions.data_cleaning(ddf_pd)
    ddf_pd = helper_functions.feature_engineering(ddf_pd)
    ddf_pd = helper_functions.add_time_features(ddf_pd)
    time3 = time.time()
    print("Feature engineering time: ", (time3 - time2)/60 , "mins")

    return ddf_pd

