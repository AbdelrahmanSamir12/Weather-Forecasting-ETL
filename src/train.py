import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
from motherduck_utils import get_connection

# Load environment variables for paths/secrets, if needed
load_dotenv(dotenv_path="../config/.env")

def load_data():
    conn = get_connection()
    df = conn.execute("SELECT * FROM processed_weather_data ORDER BY date ASC").fetchdf()
    conn.close()
    return df

def get_features_and_target(df):
    target_col = "temperature_2m"
    feature_cols = [
        # Base features
        "relative_humidity_2m", "dew_point_2m", "pressure_msl",
        "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
        # Engineered features
        "hour", "dayofweek", "month", "is_weekend", "hour_sin", "hour_cos",
        "temperature_2m_lag1", "temperature_2m_lag24", "humidity_rolling6",
        "rain_last_hour", "high_cloud", "temp_x_wind"
    ]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols

def main():
    df = load_data()
    print(f"Loaded {len(df)} rows from processed_weather_data.")

    # Clean any possible NaNs (should be done already but just in case)
    df = df.dropna().reset_index(drop=True)
    X, y, feature_cols = get_features_and_target(df)

    # Time series split: preserves order, no leakage
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_leaf=10, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        maes.append(mae)
        print(f"Fold {fold} MAE: {mae:.3f}")

    print(f"\nAverage MAE over folds: {np.mean(maes):.3f}")

    # Final model on all data
    final_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=10, 
        random_state=42, 
        n_jobs=-1
    )
    final_model.fit(X, y)

    # Save model and feature list
    os.makedirs("../data", exist_ok=True)
    joblib.dump({'model': final_model, 'features': feature_cols}, "../data/model.joblib")
    print("\nModel and feature list saved to ../data/model.joblib")

if __name__ == "__main__":
    main()