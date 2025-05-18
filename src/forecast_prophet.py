import pandas as pd
from prophet import Prophet
from prophet.serialize import  model_from_json
from src.motherduck_utils import get_connection
import os

def load_model(model_path="data/prophet_model.json"):
    with open(model_path, 'r') as fin:
        model = model_from_json(fin.read())
    return model

def make_future_df(latest_date, hours=24):
    # Prophet wants regular datetimes
    future_dates = pd.date_range(start=latest_date + pd.Timedelta(hours=1),
                                periods=hours, freq='H')
    df_future = pd.DataFrame({"ds": future_dates})
    return df_future

def forecast():
    # Get last observed date from database
    conn = get_connection()
    last_date = conn.execute("SELECT MAX(date) FROM processed_weather_data").fetchone()[0]
    conn.close()

    model = load_model()
    # Make next 24 hours of timestamps
    df_future = make_future_df(pd.to_datetime(last_date), hours=100)
    forecast = model.predict(df_future)

    # Output CSV and/or DB results
    out = forecast[["ds", "yhat"]].rename(columns={"ds": "datetime", "yhat": "temperature_2m_predicted"})
    out.to_csv("../data/prophet_forecast_24h.csv", index=False)
    print("Saved Prophet 24h forecast to ../data/prophet_forecast_24h.csv")
    print(out.head())

if __name__ == "__main__":
    forecast()