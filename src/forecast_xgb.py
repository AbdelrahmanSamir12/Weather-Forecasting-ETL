import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dotenv import load_dotenv

from src.feature_engineer import feature_engineer  # <-- This should match your ETL step!
from ingestion import fetch_weather_data

def fetch_new_api_data(lat, lon, start, end):
    # Reuse your own function from ingestion for fetching recent data
    # Or load a new CSV
    # from ingestion import fetch_weather_data
    # df = fetch_weather_data(lat, lon, start, end)
    # OR, if you already saved it as CSV:
    df = fetch_weather_data(lat,lon,start,end)
    #df = pd.read_csv("data/new_api_data.csv", parse_dates=['date'])
    return df

def main():
    load_dotenv(dotenv_path=".env")
    
    # ----------- 1. Fetch & feature engineer new data -----------
    df = fetch_new_api_data(
        lat=31.2, lon=29.9, 
        start="2025-05-17", end="2025-05-18"  # adjust range for your validation period
    )

    # ----------- 2. Engineer features as in ETL step -----------
    df_feat = feature_engineer(df)
    # Safety: Drop any possible Na
    df_feat = df_feat.dropna().reset_index(drop=True)

    # ----------- 3. Load model and apply prediction -------------
    model_record = joblib.load("models/daily_temperature_model.joblib")
    model, feature_cols = model_record["model"], model_record["features"]

    X_test = df_feat[feature_cols]
    y_true = df_feat["temperature_2m"]
    y_pred = model.predict(X_test)

    # ----------- 4. Evaluate and save results -------------------
    df_eval = pd.DataFrame({
        "datetime": df_feat["date"],
        "temperature_2m_actual": y_true,
        "temperature_2m_predicted": y_pred
    })
    df_eval.to_csv("reports/forecast_eval.csv", index=False)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    print(df_eval.head(12))

if __name__ == "__main__":
    main()