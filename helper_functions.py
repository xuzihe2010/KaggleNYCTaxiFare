import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import holidays
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

TIME_COLS = ["year", "month", "weekday", "frac_day", "is_holidays"]
FEATURE_COLS = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
                "abs_longitude", "abs_latitude", "passenger_count", "distance", "pickup_to_jfk",
                "pickup_to_ewr", "pickup_to_lgr", "pickup_to_nyc", "dropoff_to_jfk", "dropoff_to_ewr",
                "dropoff_to_lgr", "dropoff_to_nyc"] + TIME_COLS
# JFK airport coordinates, see https://www.travelmath.com/airport/JFK
JFK = (-73.7822222222, 40.6441666667)
# Newark Liberty International Airport, see https://www.travelmath.com/airport/EWR
EWR = (-74.175, 40.6897222222)
# LaGuardia Airport, see https://www.travelmath.com/airport/LGA
LGR = (-73.8719444444, 40.7747222222)
# NYC center
NYC = (-74.0063889, 40.7141667)
# Constants for calculating distance
PIE = np.pi / 180  # Pi/180
R = 6371  # Radius of the earth in km

# US Holidays
US_HOLIDAYS = holidays.US()

# Frame of predicion dataset
l_pickup_longitude, r_pickup_longitude = -74.252193 * 1.2,  -72.986532 * 0.8
l_dropoff_longitude, r_dropoff_longitude = -74.263242 * 1.2, -72.990963 * 0.8
l_pickup_latitude, r_pickup_latitude = 40.573143 * 0.8, 41.709555 * 1.2
l_dropoff_latitude, r_dropoff_latitude = 40.568973 * 0.8, 41.696683 * 1.2
l_passenger_count, r_passenger_count = 0, 8


# Data cleaning and feature engineering functions
def distance_by_long_lat(lon1, lon2, lat1, lat2):
    a = 0.5 - np.cos((lat2 - lat1) * PIE)/2 + np.cos(lat1 * PIE) * np.cos(lat2 * PIE) * (1 - np.cos((lon2 - lon1) * PIE)) / 2
    return 2 * R * np.arcsin(np.sqrt(a)) # 2*R*asin...


def minkowski_distance(x_0, x_1, y_0, y_1, p):
    return (abs(x_0 - x_1) ** p + abs(y_0 - y_1) ** p) ** (1/p)


def data_cleaning(df):
    df = df.dropna(how="any", axis="rows")
    return df[(df.pickup_longitude > l_pickup_longitude) & (df.pickup_longitude < r_pickup_longitude) &
              (df.dropoff_longitude > l_dropoff_longitude) & (df.dropoff_longitude < r_dropoff_longitude) &
              (df.pickup_latitude > l_pickup_latitude) & (df.pickup_latitude < r_pickup_latitude) &
              (df.dropoff_latitude > l_dropoff_latitude) & (df.dropoff_latitude < r_dropoff_latitude) &
              (df.passenger_count > l_passenger_count) & (df.passenger_count < r_passenger_count) &
              (df.fare_amount >= 2.5)]


def feature_engineering(df):
    df["abs_longitude"] = np.abs(df.dropoff_longitude - df.pickup_longitude)
    df["abs_latitude"] = np.abs(df.dropoff_latitude - df.pickup_latitude)
    df["distance"] = distance_by_long_lat(df.pickup_longitude, df.dropoff_longitude, df.pickup_latitude, df.dropoff_latitude)
    df["pickup_to_jfk"] = distance_by_long_lat(df.pickup_longitude, JFK[0], df.pickup_latitude, JFK[1])
    df["pickup_to_ewr"] = distance_by_long_lat(df.pickup_longitude, EWR[0], df.pickup_latitude, EWR[1])
    df["pickup_to_lgr"] = distance_by_long_lat(df.pickup_longitude, LGR[0], df.pickup_latitude, LGR[1])
    df["pickup_to_nyc"] = distance_by_long_lat(df.pickup_longitude, NYC[0], df.pickup_latitude, NYC[1])
    df["dropoff_to_jfk"] = distance_by_long_lat(df.dropoff_longitude, JFK[0], df.dropoff_latitude, JFK[1])
    df["dropoff_to_ewr"] = distance_by_long_lat(df.dropoff_longitude, EWR[0], df.dropoff_latitude, EWR[1])
    df["dropoff_to_lgr"] = distance_by_long_lat(df.dropoff_longitude, LGR[0], df.dropoff_latitude, LGR[1])
    df["dropoff_to_nyc"] = distance_by_long_lat(df.dropoff_longitude, NYC[0], df.dropoff_latitude, NYC[1])
    return df


def add_fare_bins(df):
    df["fare_bin"] = pd.cut(df.fare_amount, np.linspace(0, 50, 11)).astype(str)
    df.loc[df.fare_bin == "nan", "fare_bin"] = "(50.0+)"
    return df


def add_time_features(df):
    df["year"] = df.pickup_datetime.apply(lambda x : x.year)
    df["month"] = df.pickup_datetime.apply(lambda x : x.month)
    df["day"] = df.pickup_datetime.apply(lambda x : x.day)
    df["weekday"] = df.pickup_datetime.apply(lambda x : x.weekday())
    df["hour"] = df.pickup_datetime.apply(lambda x : x.hour)
    df["frac_day"] = (df.hour + df.pickup_datetime.apply(lambda x : x.minute) / 60 +
                      df.pickup_datetime.apply(lambda x : x.second) / 3600) / 24
    df["is_holidays"] = df.pickup_datetime.apply(lambda x : x in US_HOLIDAYS)
    return df


# Evaluation metrics and function
def metrics(train_pred, valid_pred, y_train, y_valid):
    # Root mean squared error
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    return train_rmse, valid_rmse


def evaluate(model, features, X_train, X_valid, y_train, y_valid):
    # Make predictions
    train_pred = model.predict(X_train[features])
    valid_pred = model.predict(X_valid[features])

    # Get metrics
    train_rmse, valid_rmse = metrics(train_pred, valid_pred, y_train, y_valid)

    print(f'Training:   rmse = {round(train_rmse, 3)}')
    print(f'Validation: rmse = {round(valid_rmse, 3)}')

