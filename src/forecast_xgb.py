import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from motherduck_utils import get_connection
from src.feature_engineer import feature_engineer_shiffted

# Load environment variables
load_dotenv(dotenv_path="../config/.env")

def load_latest_data(days=7):
    """Load the most recent data needed for feature engineering"""
    conn = get_connection()
    query = f"""
    SELECT * 
    FROM processed_weather_data 
    WHERE date >= CURRENT_DATE - {days+7}
    ORDER BY date ASC
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def prepare_forecast_data(model_metadata, historical_data):
    """
    Prepare the input data for forecasting by:
    1. Applying the same feature engineering as training
    2. Creating the most recent feature vector
    """
    # Apply the same preprocessing used during training
    processed_data = model_metadata['preprocess_fn'](historical_data.copy())
    
    # Get the most recent row (will use its features for prediction)
    latest_data = processed_data.iloc[[-1]].copy()
    
    # Ensure we have all required features
    required_features = model_metadata['features']
    missing_features = set(required_features) - set(latest_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    return latest_data[required_features]

def generate_forecast(model, forecast_input, days=7):
    """
    Generate multi-day forecast using iterative prediction:
    1. Predict next day's temperature
    2. Update features with the prediction
    3. Repeat for desired horizon
    """
    forecast_dates = []
    forecast_temps = []
    
    current_data = forecast_input.copy()
    
    for day in range(days):
        # Predict temperature
        temp_pred = model.predict(current_data)[0]
        pred_date = datetime.now().date() + timedelta(days=day+1)
        
        forecast_dates.append(pred_date)
        forecast_temps.append(temp_pred)
        
        # Update features for next prediction
        if day < days - 1:
            # Shift lag features
            current_data['temperature_lag1'] = temp_pred
            current_data['temperature_lag2'] = current_data['temperature_lag1']
            current_data['temperature_lag7'] = current_data['temperature_lag6'] if 'temperature_lag6' in current_data else current_data['temperature_lag1']
            
            # Note: Other features like weather conditions would need to be forecasted or assumed
            # For demo, we'll keep them constant (in practice, use weather forecasts)
    
    return pd.DataFrame({
        'date': forecast_dates,
        'predicted_temperature': forecast_temps
    })

def main():
    # Load the trained model
    model_path = "models/daily_temperature_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    model_metadata = joblib.load(model_path)
    model = model_metadata['model']
    
    # Load recent historical data
    historical_data = load_latest_data()
    if len(historical_data) < 8:  # Need at least 7 days for lag7 feature
        raise ValueError("Insufficient historical data for forecasting")
    
    # Prepare forecast input
    forecast_input = prepare_forecast_data(model_metadata, historical_data)
    
    # Generate forecast
    forecast_days = 7  # 1-week forecast
    forecast = generate_forecast(model, forecast_input, days=forecast_days)
    
    # Print results
    print("\nTemperature Forecast:")
    print("====================")
    print(f"Model MAE (estimate): {model_metadata['average_mae']:.2f}°C")
    print("\nPredicted Temperatures:")
    for _, row in forecast.iterrows():
        print(f"{row['date'].strftime('%a %b %d')}: {row['predicted_temperature']:.1f}°C")
    
    # Save forecast
    forecast.to_csv("data/latest_forecast.csv", index=False)
    print("\nForecast saved to data/latest_forecast.csv")

if __name__ == "__main__":
    main()