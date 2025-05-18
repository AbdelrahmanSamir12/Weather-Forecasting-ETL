import pandas as pd
from prophet import Prophet
from prophet.serialize  import model_to_json
from src.motherduck_utils import get_connection
import joblib
import os

def load_data():
    conn = get_connection()
    df = conn.execute("SELECT date, temperature_2m FROM processed_weather_data ORDER BY date ASC").fetchdf()
    conn.close()
    return df



def train():
    df = load_data()
    # Prophet expects column names 'ds' for date and 'y' for target
    df_prophet = df.rename(columns={"date": "ds", "temperature_2m": "y"})
    # Make sure we have no missing
    df_prophet = df_prophet.dropna()
    print(f"{len(df_prophet)} row are going to training")
    print("Sample")
    print(df.head())

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)

    #os.makedirs("data", exist_ok=True)
    #save model
    with open("data/prophet_model.json",'w') as fout:
        fout.write(model_to_json(model))


    print("Prophet model trained and saved!")

if __name__ == "__main__":
    train()
