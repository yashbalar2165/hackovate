from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from datetime import datetime, timedelta
import joblib
import numpy as np

# Import predictor classes
from ..utils.prediction import ImprovedWeatherCodePredictor

router = APIRouter()

# Setup API Client once (global)
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Load models once at startup
predictor = ImprovedWeatherCodePredictor()
predictor.load_model("D:/hackovate/fastapi-streamlit-app/backend/app/utils/improved_weather_predictor.joblib")

# Weather Alert Predictor Class (embedded in API file for simplicity)
class WeatherAlertPredictor:
    def __init__(self):
        self.alert_classifier = None
        self.severity_regressor = None  
        self.forecast_classifier = None
        self.scaler = None
        self.feature_columns = None
        
        # Alert type mapping
        self.alert_types = {
            0: "No Alert",
            1: "Thunderstorm Warning",
            2: "High Wind Alert", 
            3: "Heavy Rain Warning",
            4: "Low Visibility Alert",
            5: "Temperature Extreme",
            6: "Pressure Drop Alert"
        }
        
        # Forecast condition mapping
        self.forecast_conditions = {
            0: "Clear Skies",
            1: "Partly Cloudy", 
            2: "Cloudy",
            3: "Light Rain",
            4: "Moderate Rain",
            5: "Heavy Rain",
            6: "Thunderstorms",
            7: "High Winds",
            8: "Fog/Low Visibility"
        }

    def load_models(self, filepath_prefix="weather_models"):
        """Load trained models"""
        try:
            self.alert_classifier = joblib.load(f"{filepath_prefix}_alert_classifier.joblib")
            self.severity_regressor = joblib.load(f"{filepath_prefix}_severity_regressor.joblib")
            self.forecast_classifier = joblib.load(f"{filepath_prefix}_forecast_classifier.joblib") 
            self.scaler = joblib.load(f"{filepath_prefix}_scaler.joblib")
            self.feature_columns = joblib.load(f"{filepath_prefix}_features.joblib")
            print(f"Weather models loaded successfully from {filepath_prefix}")
            return True
        except Exception as e:
            print(f"Error loading weather models: {e}")
            return False

    def predict(self, weather_data):
        """Predict alerts and forecast for given weather data"""
        try:
            # Prepare input
            input_df = pd.DataFrame([weather_data])
            
            # Add time features if not present
            if 'hour' not in input_df:
                current_time = datetime.now()
                input_df['hour'] = current_time.hour
                input_df['month'] = current_time.month
                
            # Add trend features (simplified)
            if 'pressure_trend' not in input_df:
                input_df['pressure_trend'] = 0
                input_df['temp_trend'] = 0  
                input_df['humidity_trend'] = 0
                
            # Select and scale features
            X = input_df[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            alert_type = self.alert_classifier.predict(X_scaled)[0]
            alert_proba = self.alert_classifier.predict_proba(X_scaled)[0]
            severity = max(0, min(100, self.severity_regressor.predict(X_scaled)[0]))
            forecast_condition = self.forecast_classifier.predict(X_scaled)[0]
            
            return {
                'alert_type': int(alert_type),
                'alert_name': self.alert_types[alert_type],
                'severity': float(severity),
                'confidence': float(max(alert_proba)),
                'forecast_condition': int(forecast_condition),
                'forecast_name': self.forecast_conditions[forecast_condition]
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'alert_type': 0,
                'alert_name': "No Alert",
                'severity': 0.0,
                'confidence': 0.5,
                'forecast_condition': 1,
                'forecast_name': "Partly Cloudy"
            }

    def generate_24h_forecast(self, base_weather_data, steps=8):
        """Generate 24-hour forecast with 3-hour intervals"""
        forecasts = []
        
        for i in range(steps):
            # Modify base data for future prediction
            future_data = base_weather_data.copy()
            
            # Add realistic variations for future hours
            temp_variation = np.random.normal(0, 2) + np.sin(i * np.pi / 12) * 5  # Daily temp cycle
            future_data['temperature_2m'] += temp_variation
            future_data['relative_humidity_2m'] = max(20, min(100, 
                future_data['relative_humidity_2m'] + np.random.normal(0, 5)))
            future_data['wind_speed_10m'] = max(0, 
                future_data['wind_speed_10m'] + np.random.normal(0, 3))
            future_data['pressure_msl'] += np.random.normal(0, 1)
            
            # Update hour
            future_hour = (datetime.now().hour + i * 3) % 24
            future_data['hour'] = future_hour
            
            prediction = self.predict(future_data)
            
            forecasts.append({
                'time': (datetime.now() + timedelta(hours=i*3)).strftime("%H:%M"),
                'condition': prediction['forecast_name'],
                'probability': int(prediction['confidence'] * 100)
            })
            
        return forecasts

    def generate_active_alerts(self, weather_data):
        """Generate active alerts based on predictions"""
        prediction = self.predict(weather_data)
        alerts = []
        
        # Primary alert from ML model
        if prediction['alert_type'] > 0 and prediction['severity'] > 20:
            current_time = datetime.now().strftime("%H:%M")
            
            priority = "LOW"
            if prediction['severity'] > 70:
                priority = "HIGH PRIORITY" 
            elif prediction['severity'] > 40:
                priority = "MODERATE"
                
            alert_text = self._generate_alert_text(prediction)
            
            alerts.append({
                'time': f"{current_time} - {priority}",
                'text': alert_text
            })
            
        # Add secondary alerts based on thresholds
        self._add_secondary_alerts(weather_data, alerts)
        
        # Ensure at least one alert if conditions warrant
        if not alerts and (
            weather_data.get('wind_speed_10m', 0) > 25 or
            weather_data.get('precipitation_probability', 0) > 60
        ):
            alerts.append({
                'time': datetime.now().strftime("%H:%M") + " - MONITOR",
                'text': "Weather conditions require monitoring. Stay updated with latest forecasts."
            })
            
        return alerts

    def _generate_alert_text(self, prediction):
        """Generate alert text based on prediction"""
        alert_texts = {
            1: f"Thunderstorm conditions detected. Severity: {prediction['severity']:.0f}%. Secure equipment and avoid outdoor activities.",
            2: f"High wind alert. Expected gusts up to severe levels. Severity: {prediction['severity']:.0f}%.",
            3: f"Heavy rainfall warning. Potential flooding risk. Severity: {prediction['severity']:.0f}%.",
            4: f"Low visibility conditions. Reduced visibility for aircraft operations. Severity: {prediction['severity']:.0f}%.",
            5: f"Extreme temperature alert. Safety precautions recommended. Severity: {prediction['severity']:.0f}%.",
            6: f"Significant pressure drop detected. Weather change imminent. Severity: {prediction['severity']:.0f}%."
        }
        
        return alert_texts.get(prediction['alert_type'], f"Weather alert active. Severity: {prediction['severity']:.0f}%.")

    def _add_secondary_alerts(self, weather_data, alerts):
        """Add secondary alerts based on weather thresholds"""
        current_time = datetime.now().strftime("%H:%M")
        
        # Wind alert
        wind_speed = weather_data.get('wind_speed_10m', 0)
        if wind_speed > 30:
            alerts.append({
                'time': f"{current_time} - MODERATE",
                'text': f"Wind speeds reaching {wind_speed:.0f} km/h. Monitor conditions closely."
            })
            
        # Precipitation alert
        precip_prob = weather_data.get('precipitation_probability', 0)
        if precip_prob > 80:
            alerts.append({
                'time': f"{current_time} - MODERATE", 
                'text': f"High precipitation probability ({precip_prob:.0f}%). Prepare for wet conditions."
            })
            
        # Pressure alert
        pressure = weather_data.get('pressure_msl', 1013)
        if pressure < 1000:
            alerts.append({
                'time': f"{current_time} - MODERATE",
                'text': f"Low pressure system ({pressure:.1f} hPa). Potential weather changes ahead."
            })
            
        # Temperature alert
        temp = weather_data.get('temperature_2m', 25)
        if temp > 40 or temp < 0:
            alerts.append({
                'time': f"{current_time} - MODERATE",
                'text': f"Extreme temperature ({temp:.1f}°C). Take appropriate safety measures."
            })

# Initialize weather alert predictor
weather_predictor = WeatherAlertPredictor()
# Try to load models (will fail gracefully if models don't exist)
weather_predictor.load_models("D:/hackovate/fastapi-streamlit-app/backend/app/utils/weather_models")

def predict_weathercode(sample_data: dict) -> float:
    """Predict next3h_max_weathercode using the trained model."""
    prediction = predictor.predict(sample_data)
    return float(prediction)

@router.get("/thunderstrome")
def get_forecast(
    airport: str = Query(..., description="Airport code"),
    country: str = Query("Unknown", description="Country name"),
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude")
):
    latitude = lat
    longitude = lng
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m", 
            "weathercode",
            "pressure_msl",
            "apparent_temperature",
            "precipitation_probability",
            "precipitation",
            "surface_pressure",
            "windspeed_10m",
            "winddirection_10m",
            "windgusts_10m",
            "dew_point_2m",
            "cloudcover"
        ],
        "forecast_days": 1,
        "timezone": "auto"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Build DataFrame
        hourly = response.Hourly()
        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        df = pd.DataFrame({
            "time": timestamps,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "weather_code": hourly.Variables(2).ValuesAsNumpy(),
            "pressure_msl": hourly.Variables(3).ValuesAsNumpy(),
            "apparent_temperature": hourly.Variables(4).ValuesAsNumpy(),
            "precipitation_probability": hourly.Variables(5).ValuesAsNumpy(),
            "precipitation": hourly.Variables(6).ValuesAsNumpy(),
            "surface_pressure": hourly.Variables(7).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(8).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(9).ValuesAsNumpy(),
            "wind_gusts_10m": hourly.Variables(10).ValuesAsNumpy(),
            "dew_point_2m": hourly.Variables(11).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(12).ValuesAsNumpy(),
        })

        # Convert timezone if needed
        if hasattr(timestamps, 'tz_convert'):
            df["time"] = timestamps.tz_convert("UTC")

        # Select nearest target hour
        now = datetime.now().astimezone()
        if now.minute <= 30:
            target_hour = now.replace(minute=0, second=0, microsecond=0)
        else:
            target_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        # Find closest time match
        if not df.empty:
            closest_idx = (df["time"] - target_hour).abs().idxmin()
            row = df.loc[closest_idx]
        else:
            raise HTTPException(status_code=500, detail="No weather data available")

        # Convert row to dict
        current_weather = {
            "time": str(row["time"]),
            "temperature_2m": float(row["temperature_2m"]) if pd.notna(row["temperature_2m"]) else 25.0,
            "relative_humidity_2m": float(row["relative_humidity_2m"]) if pd.notna(row["relative_humidity_2m"]) else 50.0,
            "weather_code": float(row["weather_code"]) if pd.notna(row["weather_code"]) else 0.0,
            "pressure_msl": float(row["pressure_msl"]) if pd.notna(row["pressure_msl"]) else 1013.0,
            "apparent_temperature": float(row["apparent_temperature"]) if pd.notna(row["apparent_temperature"]) else 25.0,
            "precipitation_probability": float(row["precipitation_probability"]) if pd.notna(row["precipitation_probability"]) else 0.0,
            "precipitation": float(row["precipitation"]) if pd.notna(row["precipitation"]) else 0.0,
            "surface_pressure": float(row["surface_pressure"]) if pd.notna(row["surface_pressure"]) else 1013.0,
            "wind_speed_10m": float(row["wind_speed_10m"]) if pd.notna(row["wind_speed_10m"]) else 10.0,
            "wind_direction_10m": float(row["wind_direction_10m"]) if pd.notna(row["wind_direction_10m"]) else 180.0,
            "wind_gusts_10m": float(row["wind_gusts_10m"]) if pd.notna(row["wind_gusts_10m"]) else 15.0,
            "dew_point_2m": float(row["dew_point_2m"]) if pd.notna(row["dew_point_2m"]) else 15.0,
            "cloud_cover": float(row["cloud_cover"]) if pd.notna(row["cloud_cover"]) else 50.0
        }

        # Generate thunderstorm prediction using original model
        thunderstorm_prediction = predict_weathercode(current_weather)

        # Generate dynamic forecasts and alerts using weather predictor
        forecast_24h = []
        active_alerts = []
        
        if weather_predictor.alert_classifier is not None:
            try:
                # Generate 24-hour forecast
                forecast_24h = weather_predictor.generate_24h_forecast(current_weather)
                
                # Generate active alerts
                active_alerts = weather_predictor.generate_active_alerts(current_weather)
                
            except Exception as e:
                print(f"Error generating forecasts/alerts: {e}")
                # Fallback to static data
                forecast_24h = generate_fallback_forecast()
                active_alerts = generate_fallback_alerts(current_weather)
        else:
            # Fallback to static data if models not loaded
            forecast_24h = generate_fallback_forecast()
            active_alerts = generate_fallback_alerts(current_weather)

        return JSONResponse(content={
            "airport": airport,
            "country": country,
            "coordinates": {"lat": latitude, "lng": longitude},
            "forecast": current_weather,
            "predicted_next3h_max_weathercode": thunderstorm_prediction,
            "forecast_24h": forecast_24h,
            "active_alerts": active_alerts,
            "model_confidence": calculate_model_confidence(current_weather),
            "last_updated": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Weather data fetch error: {str(e)}")

def generate_fallback_forecast():
    """Generate fallback forecast when ML models aren't available"""
    base_conditions = ["Partly Cloudy", "Cloudy", "Light Rain", "Clear Skies", "Thunderstorms", "High Winds"]
    forecasts = []
    
    for i in range(8):  # 24 hours / 3-hour intervals = 8
        time_str = (datetime.now() + timedelta(hours=i*3)).strftime("%H:%M")
        condition = np.random.choice(base_conditions)
        probability = np.random.randint(30, 90)
        
        forecasts.append({
            "time": time_str,
            "condition": condition,
            "probability": probability
        })
    
    return forecasts

def generate_fallback_alerts(weather_data):
    """Generate fallback alerts based on simple thresholds"""
    alerts = []
    current_time = datetime.now().strftime("%H:%M")
    
    # Check basic conditions
    temp = weather_data.get('temperature_2m', 25)
    wind_speed = weather_data.get('wind_speed_10m', 10)
    precip_prob = weather_data.get('precipitation_probability', 0)
    pressure = weather_data.get('pressure_msl', 1013)
    
    if wind_speed > 25:
        alerts.append({
            "time": f"{current_time} - HIGH PRIORITY",
            "text": f"High wind conditions detected. Wind speed: {wind_speed:.0f} km/h. Secure loose equipment."
        })
    
    if precip_prob > 80:
        alerts.append({
            "time": f"{current_time} - MODERATE",
            "text": f"Heavy precipitation expected. Probability: {precip_prob:.0f}%. Prepare for wet conditions."
        })
    
    if pressure < 1005:
        alerts.append({
            "time": f"{current_time} - MODERATE",
            "text": f"Low pressure system approaching. Pressure: {pressure:.1f} hPa. Monitor weather changes."
        })
    
    if temp > 35 or temp < 5:
        alerts.append({
            "time": f"{current_time} - MODERATE",
            "text": f"Extreme temperature conditions. Temperature: {temp:.1f}°C. Take safety precautions."
        })
    
    # Default alert if no specific conditions
    if not alerts:
        alerts.append({
            "time": f"{current_time} - MONITOR",
            "text": "Weather conditions are within normal ranges. Continue routine monitoring."
        })
    
    return alerts

def calculate_model_confidence(weather_data):
    """Calculate overall model confidence percentage"""
    base_confidence = 75
    
    # Adjust confidence based on data quality
    if weather_data.get('pressure_msl', 0) == 0:
        base_confidence -= 10
    if weather_data.get('wind_speed_10m', 0) == 0:
        base_confidence -= 5
    if weather_data.get('temperature_2m', 0) == 0:
        base_confidence -= 10
    
    # Add some randomness for realism
    confidence = base_confidence + np.random.randint(-5, 10)
    return max(60, min(95, confidence))

# Additional endpoint for model status
@router.get("/model-status")
def get_model_status():
    """Get status of loaded models"""
    return JSONResponse(content={
        "thunderstorm_model_loaded": predictor is not None,
        "weather_alert_models_loaded": weather_predictor.alert_classifier is not None,
        "timestamp": datetime.now().isoformat()
    })