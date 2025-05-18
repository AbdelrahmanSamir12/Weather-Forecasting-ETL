from src.ingestion import fetch_weather_data,store_to_motherduck
from src.etl import preprocess_and_store
from src.train_prophet import load_data,train
from src.forecast_prophet import forecast

from src.motherduck_utils import get_connection, create_tables



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
        start_date="2020-05-08",
        end_date="2025-05-16"
    )
    
    # For local backup
    df.to_csv("weather_data.csv", index=False)

    # For ML pipeline
    store_to_motherduck(df)


    #ETL
    preprocess_and_store()

    # Train 
    train()

    # forecast 
    forecast()







