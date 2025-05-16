import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os

from motherduck_utils import get_connection, create_tables  # <-- Use your utility

def fetch_weather_data(
    latitude,
    longitude,
    start_date,
    end_date,
    variables=["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl", "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
) -> pd.DataFrame:
    # Setup API Client
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables)
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Build data dict
    hourly = response.Hourly()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    for idx, var in enumerate(variables):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    df = pd.DataFrame(data)
    return df

def store_to_motherduck(df: pd.DataFrame):
    conn = get_connection()
    create_tables(conn)
    # Avoid inserting duplicates (up to you to deduplicate based on project needs)
    conn.register("df", df)
    #delete previuos
    conn.execute("DELETE FROM processed_weather_data")
    conn.execute("""
        INSERT INTO raw_weather_data
        SELECT * FROM df
    """)
    conn.close()
    print(f"Inserted {len(df)} rows to MotherDuck raw_weather_data table")

if __name__ == "__main__":

    cairo = {
        "lat" :30.0444 ,
        "long" : 31.2357
    }
    Alexandria = {
        "lat" :31.2001 ,
        "long" : 29.9187
    }
    df = fetch_weather_data(
        latitude=30.0444,
        longitude=31.2357,
        start_date="2024-05-08",
        end_date="2025-05-16"
    )
    
    # For local backup
    df.to_csv("weather_data.csv", index=False)

    # For ML pipeline
    store_to_motherduck(df)