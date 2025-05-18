from src.motherduck_utils import get_connection
from src.feature_engineer import feature_engineer_shiffted, feature_engineer
def preprocess_and_store():
    conn = get_connection()

    # Read raw data
    df_raw = conn.execute("SELECT * FROM raw_weather_data").fetchdf()

    # Datetime Features: Extract Temporal Patterns

    df = feature_engineer_shiffted(df_raw)
    # df['hour'] = df['date'].dt.hour
    # df['dayofweek'] = df['date'].dt.dayofweek   # Monday=0
    # df['month'] = df['date'].dt.month
    # df['is_weekend'] = df['date'].dt.dayofweek >= 5

    # # To encode hourly periodicity (important for daily cycles)
    # import numpy as np
    # df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    # df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)


    # #Lag 
    # # previous hour temperature as a feature
    # df['temperature_2m_lag1'] = df['temperature_2m'].shift(1)
    # df['temperature_2m_lag2'] = df['temperature_2m'].shift(2)
    # df['temperature_2m_lag24'] = df['temperature_2m'].shift(24)   # previous day's same hour

    # # Do the same for dew_point or humidity if useful:
    # df['dew_point_2m_lag1'] = df['dew_point_2m'].shift(1)

    # # Rolling Avergae
    # # past 3-hour moving average
    # df['temp_rolling3'] = df['temperature_2m'].rolling(window=3).mean()
    # df['humidity_rolling6'] = df['relative_humidity_2m'].rolling(window=6).mean()
    # df['wind_speed_rolling3'] = df['wind_speed_10m'].rolling(window=3).mean()

    # #Binary Weather condition flag
    # # Rain in the last hour
    # df['rain_last_hour'] = (df['precipitation'] > 0).astype(int)
    # # High cloud cover
    # df['high_cloud'] = (df['cloud_cover'] > 50).astype(int)

    # # interactions
    # df['temp_x_wind'] = df['temperature_2m'] * df['wind_speed_10m']

    # # drop nans
    # df = df.dropna().reset_index(drop=True)


    # Save to processed table
    conn.register("processed_df", df)
    conn.execute("DELETE FROM processed_weather_data")  # clear old
    conn.execute("""
        INSERT INTO processed_weather_data 
        SELECT * FROM processed_df
    """)
    # conn.execute("""
    # CREATE TABLE IF NOT EXISTS processed_weather_data AS 
    #     SELECT * FROM processed_df WHERE FALSE;
    # """)
    conn.close()

if __name__ == "__main__":
    preprocess_and_store()