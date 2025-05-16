import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
from motherduck_utils import get_connection

# Load environment variables
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

    # Clean any possible NaNs
    df = df.dropna().reset_index(drop=True)
    X, y, feature_cols = get_features_and_target(df)

    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.7,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mae"
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        maes.append(mae)
        print(f"Fold {fold} MAE: {mae:.3f}")

    print(f"\nAverage MAE over folds: {np.mean(maes):.3f}")

    # Train on all data
    final_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.7,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    final_model.fit(X, y)

    #os.makedirs("../data", exist_ok=True)
    joblib.dump({'model': final_model, 'features': feature_cols}, "models/model.joblib")
    print("\nXGBoost model and feature list saved to models/model.joblib")

if __name__ == "__main__":
    main()