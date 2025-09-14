import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WeatherAlertPredictor:
    def __init__(self):
        # Alert classifier (predicts alert type)
        self.alert_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        # Severity regressor (predicts alert severity 0-100)
        self.severity_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        # Forecast classifier (predicts weather conditions for next 24h)
        self.forecast_classifier = RandomForestClassifier(n_estimators=150, random_state=42)
        
        self.scaler = StandardScaler()
        self.feature_columns = [
            'temperature_2m', 'relative_humidity_2m', 'pressure_msl', 
            'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
            'precipitation', 'precipitation_probability', 'cloud_cover',
            'dew_point_2m', 'apparent_temperature', 'surface_pressure',
            'hour', 'month', 'pressure_trend', 'temp_trend', 'humidity_trend'
        ]
        
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

    def generate_synthetic_data(self, n_samples=50000):
        """Generate synthetic weather data for training"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Base weather parameters
            temp = np.random.normal(25, 15)  # Temperature
            humidity = np.random.uniform(20, 95)  # Humidity
            pressure = np.random.normal(1013, 20)  # Pressure
            wind_speed = np.random.exponential(15)  # Wind speed
            wind_direction = np.random.uniform(0, 360)  # Wind direction
            wind_gusts = wind_speed * np.random.uniform(1.2, 2.5)  # Wind gusts
            precipitation = np.random.exponential(2) if np.random.random() > 0.7 else 0
            precip_prob = min(100, precipitation * 20 + np.random.normal(0, 10))
            cloud_cover = np.random.uniform(0, 100)
            dew_point = temp - (100 - humidity) / 5  # Simplified dew point
            apparent_temp = temp + humidity/100 * 2 - wind_speed/10
            surface_pressure = pressure + np.random.normal(0, 5)
            
            # Time features
            hour = np.random.randint(0, 24)
            month = np.random.randint(1, 13)
            
            # Trends (simulated)
            pressure_trend = np.random.normal(0, 2)
            temp_trend = np.random.normal(0, 1) 
            humidity_trend = np.random.normal(0, 3)
            
            # Generate alerts based on weather conditions
            alert_type = 0  # No alert by default
            severity = 0
            
            # Thunderstorm conditions
            if (pressure < 1005 and humidity > 75 and temp > 20 and 
                wind_speed > 25 and cloud_cover > 70):
                alert_type = 1
                severity = min(100, 40 + wind_speed + (80 - pressure)/2)
                
            # High wind conditions  
            elif wind_speed > 40 or wind_gusts > 60:
                alert_type = 2
                severity = min(100, wind_speed + wind_gusts/2 - 20)
                
            # Heavy rain conditions
            elif precipitation > 10 and precip_prob > 80:
                alert_type = 3 
                severity = min(100, precipitation * 5 + precip_prob/2)
                
            # Low visibility conditions
            elif cloud_cover > 90 and humidity > 85 and wind_speed < 5:
                alert_type = 4
                severity = min(100, cloud_cover + humidity - 80)
                
            # Temperature extremes
            elif temp > 45 or temp < -10:
                alert_type = 5
                severity = min(100, abs(temp - 25) * 2)
                
            # Pressure drop
            elif pressure < 995 and pressure_trend < -3:
                alert_type = 6
                severity = min(100, (1013 - pressure) + abs(pressure_trend) * 10)
                
            # Generate forecast conditions
            forecast_condition = 0  # Clear by default
            
            if precipitation > 15:
                forecast_condition = 6  # Thunderstorms
            elif precipitation > 5:
                forecast_condition = 5 if wind_speed > 20 else 4  # Heavy/Moderate rain
            elif precipitation > 0:
                forecast_condition = 3  # Light rain
            elif wind_speed > 35:
                forecast_condition = 7  # High winds
            elif cloud_cover < 20:
                forecast_condition = 0  # Clear
            elif cloud_cover < 60:
                forecast_condition = 1  # Partly cloudy
            elif humidity > 85 and wind_speed < 5:
                forecast_condition = 8  # Fog
            else:
                forecast_condition = 2  # Cloudy
                
            data.append({
                'temperature_2m': temp,
                'relative_humidity_2m': humidity, 
                'pressure_msl': pressure,
                'wind_speed_10m': wind_speed,
                'wind_direction_10m': wind_direction,
                'wind_gusts_10m': wind_gusts,
                'precipitation': precipitation,
                'precipitation_probability': precip_prob,
                'cloud_cover': cloud_cover,
                'dew_point_2m': dew_point,
                'apparent_temperature': apparent_temp,
                'surface_pressure': surface_pressure,
                'hour': hour,
                'month': month,
                'pressure_trend': pressure_trend,
                'temp_trend': temp_trend,
                'humidity_trend': humidity_trend,
                'alert_type': alert_type,
                'severity': max(0, severity),
                'forecast_condition': forecast_condition
            })
            
        return pd.DataFrame(data)

    def prepare_features(self, df):
        """Prepare features for training"""
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X

    def train(self, df):
        """Train all models"""
        print("Preparing features...")
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare targets
        y_alert = df['alert_type']
        y_severity = df['severity'] 
        y_forecast = df['forecast_condition']
        
        # Split data
        X_train, X_test, y_alert_train, y_alert_test = train_test_split(
            X_scaled, y_alert, test_size=0.2, random_state=42, stratify=y_alert
        )
        
        _, _, y_sev_train, y_sev_test = train_test_split(
            X_scaled, y_severity, test_size=0.2, random_state=42
        )
        
        _, _, y_fore_train, y_fore_test = train_test_split(
            X_scaled, y_forecast, test_size=0.2, random_state=42, stratify=y_forecast
        )
        
        print("Training alert classifier...")
        self.alert_classifier.fit(X_train, y_alert_train)
        alert_pred = self.alert_classifier.predict(X_test)
        print(f"Alert Classification Accuracy: {accuracy_score(y_alert_test, alert_pred):.3f}")
        
        print("Training severity regressor...")
        self.severity_regressor.fit(X_train, y_sev_train)
        sev_pred = self.severity_regressor.predict(X_test)
        print(f"Severity R² Score: {r2_score(y_sev_test, sev_pred):.3f}")
        
        print("Training forecast classifier...")
        self.forecast_classifier.fit(X_train, y_fore_train)
        fore_pred = self.forecast_classifier.predict(X_test)
        print(f"Forecast Classification Accuracy: {accuracy_score(y_fore_test, fore_pred):.3f}")

    def predict(self, weather_data):
        """Predict alerts and forecast for given weather data"""
        # Prepare input
        input_df = pd.DataFrame([weather_data])
        
        # Add time features if not present
        if 'hour' not in input_df:
            current_time = datetime.now()
            input_df['hour'] = current_time.hour
            input_df['month'] = current_time.month
            
        # Add trend features (simplified - would be calculated from historical data)
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

    def generate_24h_forecast(self, base_weather_data, steps=8):
        """Generate 24-hour forecast with 3-hour intervals"""
        forecasts = []
        
        for i in range(steps):
            # Modify base data for future prediction
            future_data = base_weather_data.copy()
            
            # Add some randomness and trends for future hours
            future_data['temperature_2m'] += np.random.normal(0, 2) + i * 0.5
            future_data['relative_humidity_2m'] += np.random.normal(0, 5)  
            future_data['wind_speed_10m'] += np.random.normal(0, 3)
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
            
        # Add secondary alerts based on conditions
        self._add_secondary_alerts(weather_data, alerts)
        
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
        
        if weather_data.get('wind_speed_10m', 0) > 30:
            alerts.append({
                'time': f"{current_time} - MODERATE",
                'text': f"Wind speeds reaching {weather_data['wind_speed_10m']:.0f} km/h. Monitor conditions closely."
            })
            
        if weather_data.get('precipitation_probability', 0) > 80:
            alerts.append({
                'time': f"{current_time} - MODERATE", 
                'text': f"High precipitation probability ({weather_data['precipitation_probability']:.0f}%). Prepare for wet conditions."
            })

    def save_models(self, filepath_prefix="weather_models"):
        """Save trained models"""
        joblib.dump(self.alert_classifier, f"{filepath_prefix}_alert_classifier.joblib")
        joblib.dump(self.severity_regressor, f"{filepath_prefix}_severity_regressor.joblib") 
        joblib.dump(self.forecast_classifier, f"{filepath_prefix}_forecast_classifier.joblib")
        joblib.dump(self.scaler, f"{filepath_prefix}_scaler.joblib")
        joblib.dump(self.feature_columns, f"{filepath_prefix}_features.joblib")
        print(f"Models saved with prefix: {filepath_prefix}")

    def load_models(self, filepath_prefix="weather_models"):
        """Load trained models"""
        self.alert_classifier = joblib.load(f"{filepath_prefix}_alert_classifier.joblib")
        self.severity_regressor = joblib.load(f"{filepath_prefix}_severity_regressor.joblib")
        self.forecast_classifier = joblib.load(f"{filepath_prefix}_forecast_classifier.joblib") 
        self.scaler = joblib.load(f"{filepath_prefix}_scaler.joblib")
        self.feature_columns = joblib.load(f"{filepath_prefix}_features.joblib")
        print(f"Models loaded from prefix: {filepath_prefix}")

def main():
    """Main training script"""
    print("Initializing Weather Alert Predictor...")
    predictor = WeatherAlertPredictor()
    
    print("Generating synthetic training data...")
    df = predictor.generate_synthetic_data(n_samples=50000)
    print(f"Generated {len(df)} training samples")
    
    print("Training models...")
    predictor.train(df)
    
    print("Saving models...")
    predictor.save_models("weather_models")
    
    print("Training completed successfully!")
    
    # Test prediction
    print("\nTesting prediction...")
    sample_data = {
        'temperature_2m': 28.5,
        'relative_humidity_2m': 85,
        'pressure_msl': 1005.2,
        'wind_speed_10m': 35,
        'wind_direction_10m': 220,
        'wind_gusts_10m': 45,
        'precipitation': 5.2,
        'precipitation_probability': 85,
        'cloud_cover': 75,
        'dew_point_2m': 25.1,
        'apparent_temperature': 32.1,
        'surface_pressure': 1006.8
    }
    
    result = predictor.predict(sample_data)
    print("Sample prediction result:")
    print(f"  Alert: {result['alert_name']} (Severity: {result['severity']:.1f}%)")
    print(f"  Forecast: {result['forecast_name']} (Confidence: {result['confidence']:.2f})")

if __name__ == "__main__":
    main()