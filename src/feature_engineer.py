import pandas as pd
import numpy as np

def feature_engineer(df):
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek   # Monday=0
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek >= 5

    # To encode hourly periodicity (important for daily cycles)
    import numpy as np
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)


    #Lag 
    # previous hour temperature as a feature
    df['temperature_2m_lag1'] = df['temperature_2m'].shift(1)
    df['temperature_2m_lag2'] = df['temperature_2m'].shift(2)
    df['temperature_2m_lag24'] = df['temperature_2m'].shift(24)   # previous day's same hour

    # Do the same for dew_point or humidity if useful:
    df['dew_point_2m_lag1'] = df['dew_point_2m'].shift(1)

    # Rolling Avergae
    # past 3-hour moving average
    df['temp_rolling3'] = df['temperature_2m'].rolling(window=3).mean()
    df['humidity_rolling6'] = df['relative_humidity_2m'].rolling(window=6).mean()
    df['wind_speed_rolling3'] = df['wind_speed_10m'].rolling(window=3).mean()

    #Binary Weather condition flag
    # Rain in the last hour
    df['rain_last_hour'] = (df['precipitation'] > 0).astype(int)
    # High cloud cover
    df['high_cloud'] = (df['cloud_cover'] > 50).astype(int)

    # interactions
    df['temp_x_wind'] = df['temperature_2m'] * df['wind_speed_10m']

    # drop nans
    return df.dropna().reset_index(drop=True)

def feature_engineer_shiffted(df):
    # First resample to daily data
    df_daily = df.resample('D', on='date').agg({
        'temperature_2m': 'mean',
        'dew_point_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'wind_speed_10m': 'mean',
        'precipitation': 'sum',
        'cloud_cover': 'mean'
    }).reset_index()
    
    # Create time features from date
    # df_daily['dayofweek'] = df_daily['date'].dt.dayofweek   # Monday=0
    # df_daily['month'] = df_daily['date'].dt.month
    # df_daily['is_weekend'] = df_daily['date'].dt.dayofweek >= 5
    # df_daily['day'] = df_daily['date'].dt.day
    
    # To encode yearly periodicity

    df_daily['dayofyear_sin'] = np.sin(2 * np.pi * df_daily['date'].dt.dayofyear/365)
    df_daily['dayofyear_cos'] = np.cos(2 * np.pi * df_daily['date'].dt.dayofyear/365)
    
    
    
    # Lag features - previous days' temperatures
    df_daily['temperature_lag1'] = df_daily['temperature_2m'].shift(1)  # yesterday
    df_daily['temperature_lag2'] = df_daily['temperature_2m'].shift(2)  # 2 days ago
    df_daily['temperature_lag7'] = df_daily['temperature_2m'].shift(7)  # same day last week
    
    # Rolling averages
    # df_daily['temp_rolling3'] = df_daily['temperature_2m'].rolling(window=3).mean()  # 3-day moving avg
    # df_daily['temp_rolling7'] = df_daily['temperature_2m'].rolling(window=7).mean()  # weekly moving avg
    
    # Weather condition flags
    df_daily['had_precipitation'] = (df_daily['precipitation'] > 0).astype(int)  # rain/snow today
    df_daily['high_cloud'] = (df_daily['cloud_cover'] > 50).astype(int)
    
    # Interactions
    df_daily['temp_x_wind'] = df_daily['temperature_2m'] * df_daily['wind_speed_10m']

    # Create target - tomorrow's temperature (shift temperature back by 1 day)
    df_daily['temperature_2m'] = df_daily['temperature_2m'].shift(-1)
    
    # Drop rows with missing values (including the last row which has no target)
    return df_daily.dropna().reset_index(drop=True)
