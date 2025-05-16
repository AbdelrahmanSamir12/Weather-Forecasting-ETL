import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os

def fetch_weather_data(
    latitude,
    longitude,
    start_date,
    end_date,
    variables=["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl", "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
    save_path="weather_data.csv"
):
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
    # For local time, you can convert based on timezone if needed
    # df['local_time'] = df['date'].dt.tz_convert(response.Timezone())

    df.to_csv(save_path, index=False)
    print(f"Saved data to {os.path.abspath(save_path)}")
    print(df.head())

# Usage Example:
fetch_weather_data(
    latitude=31.2001,
    longitude=29.9187,
    start_date="2025-04-25",
    end_date="2025-05-08"
)