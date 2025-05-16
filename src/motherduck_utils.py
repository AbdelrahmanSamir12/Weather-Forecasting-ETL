import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f"md:?motherduck_token={token}")

def create_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_weather_data (
            date TIMESTAMP,
            temperature_2m FLOAT,
            relative_humidity_2m FLOAT,
            dew_point_2m FLOAT,
            pressure_msl FLOAT,
            precipitation FLOAT,
            cloud_cover FLOAT,
            wind_speed_10m FLOAT,
            wind_direction_10m FLOAT
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_weather_data AS 
            SELECT * FROM raw_weather_data WHERE FALSE;
    """)


    
