import os

# Configurations
RAW_DATA_TABLE = "raw_weather_data"
PROCESSED_DATA_TABLE = "processed_weather_data"
MODEL_PATH = os.getenv("MODEL_PATH", "data/prophet_model.json")
FORECAST_OUTPUT_PATH = os.getenv("FORECAST_OUTPUT_PATH", "data/prophet_forecast_24h.csv")
DEFAULT_LOCATIONS = [
    {"city": "Cairo", "lat": 30.0444, "lon": 31.2357},
    {"city": "Alexandria", "lat": 31.2001, "lon": 29.9187},
]