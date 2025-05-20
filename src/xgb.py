import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
from motherduck_utils import get_connection
from src.feature_engineer import feature_engineer_shiffted
# Load environment variables
load_dotenv(dotenv_path="../config/.env")

def load_data():
    conn = get_connection()
    df = conn.execute("SELECT * FROM processed_weather_data ORDER BY date ASC").fetchdf()
    conn.close()
    return df


def get_features_and_target(df):
    target_col = "target_temperature"  # This is tomorrow's temperature
    feature_cols = [
        # Base features
        "dew_point_2m", "relative_humidity_2m",
        "wind_speed_10m", "precipitation", "cloud_cover",
        
        # Time features
        "dayofyear_sin", "dayofyear_cos",
        
        # Engineered features
        "temperature_lag1", "temperature_lag2", "temperature_lag7",
        #"temp_rolling3", "temp_rolling7",
        "had_precipitation", "high_cloud", "temp_x_wind"
    ]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols

def main():
    df = load_data()
    print(df.info())
    print(f"Loaded {len(df)} rows from processed_weather_data.")
    

    X, y, feature_cols = get_features_and_target(df)

    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(
            n_estimators=300,  # Slightly more trees for daily data
            max_depth=5,      # Slightly shallower to prevent overfitting
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mae",
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10  # Print every 10 iterations
        )
        
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        maes.append(mae)
        models.append(model)
        print(f"Fold {fold} MAE: {mae:.3f}°C")
        print(f"Fold {fold} Feature importances:")
        for name, importance in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
            print(f"  {name}: {importance:.3f}")

    print(f"\nAverage MAE over folds: {np.mean(maes):.3f}°C")

    # Train final model on all data using best parameters
    final_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    final_model.fit(X, y)

    # Save model and metadata
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        'model': final_model,
        'features': feature_cols,
        'average_mae': np.mean(maes),
        'preprocess_fn': feature_engineer_shiffted
    }, "models/daily_temperature_model.joblib")
    
    print("\nModel saved to models/daily_temperature_model.joblib")
    print(f"Final model MAE estimate: {np.mean(maes):.3f}°C")

if __name__ == "__main__":
    main()