from prediction import ImprovedWeatherCodePredictor  # or just import from same file if no package
# If your training code is in train_model.py, then:
# from train_model import ImprovedWeatherCodePredictor

# Step 1: Load trained model
predictor = ImprovedWeatherCodePredictor()
predictor.load_model("improved_weather_predictor.joblib")

# Step 2: Prepare new input data
sample_data = {
        'time': '2025-09-13T15:00',
        'temperature_2m': 32.1,
        'relative_humidity_2m': 61,
        'weather_code': 51,
        'pressure_msl': 1000.1,
        'apparent_temperature': 34.6,
        'precipitation_probability': 1,
        'precipitation': 0.1,
        'surface_pressure': 991.3,
        'wind_speed_10m': 28.0,
        'wind_direction_10m': 228,
        'wind_gusts_10m': 30.2,
        'dew_point_2m': 23.7,
        'cloud_cover': 100
    }

# Step 3: Predict using trained model
prediction = predictor.predict(sample_data)

print(f"Predicted next3h_max_weathercode: {prediction:.1f}")
