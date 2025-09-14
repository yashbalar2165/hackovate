import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

class ImprovedWeatherCodePredictor:
    """
    Improved XGBoost model to predict next3h_max_weathercode (0-100)
    Addresses overfitting and improves generalization
    """
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.feature_columns = []
        self.target_column = 'next3h_max_weathercode'
        self.feature_importance_ = None
        
    def load_and_prepare_data(self, csv_file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load CSV data and prepare features with better feature engineering"""
        
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} rows from {csv_file_path}")
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')  # Ensure chronological order
        
        # Clean column names
        df.columns = [col.split(' (')[0] for col in df.columns]
        
        # Enhanced feature engineering
        self._create_enhanced_features(df)
        
        # Select meaningful features (avoid too many to prevent overfitting)
        self.feature_columns = self._select_core_features(df)
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Better handling of missing values
        X = self._handle_missing_values(X)
        y = y.fillna(y.median())
        
        # Remove invalid rows
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Final dataset: {len(X)} samples, {len(self.feature_columns)} features")
        print(f"Target distribution - Min: {y.min():.1f}, Max: {y.max():.1f}, Mean: {y.mean():.1f}, Std: {y.std():.1f}")
        
        return X, y
    
    def _create_enhanced_features(self, df: pd.DataFrame):
        """Create meaningful weather features without over-engineering"""
        
        # Time-based features (important for weather patterns)
        df['hour'] = df['time'].dt.hour
        df['day_of_year'] = df['time'].dt.dayofyear
        df['month'] = df['time'].dt.month
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        df['season'] = (df['month'] % 12 // 3).astype(int)  # 0=winter, 1=spring, etc.
        
        # Key weather relationships
        df['temp_dewpoint_spread'] = df['temperature_2m'] - df['dew_point_2m']
        df['temp_apparent_diff'] = df['temperature_2m'] - df['apparent_temperature']
        df['pressure_surface_diff'] = df['pressure_msl'] - df['surface_pressure']
        
        # Wind characteristics
        df['wind_gust_ratio'] = df['wind_gusts_10m'] / (df['wind_speed_10m'] + 1)
        
        # Atmospheric stability indicators
        df['instability_indicator'] = (
            (df['temp_dewpoint_spread'] < 5) & 
            (df['relative_humidity_2m'] > 75)
        ).astype(int)
        
        # Weather severity mapping based on WMO codes
        weather_severity_map = {
            0: 0, 1: 5, 2: 10, 3: 15,  # Clear to overcast
            45: 25, 48: 30,  # Fog
            51: 20, 53: 25, 55: 30, 56: 35, 57: 40,  # Drizzle
            61: 30, 63: 40, 65: 50, 66: 55, 67: 60,  # Rain
            71: 35, 73: 45, 75: 55, 77: 50,  # Snow
            80: 40, 81: 50, 82: 60, 85: 55, 86: 65,  # Showers
            95: 75, 96: 85, 99: 95  # Thunderstorms
        }
        df['current_weather_severity'] = df['weather_code'].map(weather_severity_map).fillna(15)
        
        # Short-term trends (only 1-3 hour windows to avoid overfitting)
        df['pressure_trend_1h'] = df['pressure_msl'].diff(1)
        df['temp_trend_1h'] = df['temperature_2m'].diff(1)
        df['humidity_trend_1h'] = df['relative_humidity_2m'].diff(1)
        df['wind_trend_1h'] = df['wind_speed_10m'].diff(1)
        
        # 3-hour moving averages (smooth out noise)
        for col in ['pressure_msl', 'temperature_2m', 'wind_speed_10m']:
            if col in df.columns:
                df[f'{col}_3h_mean'] = df[col].rolling(window=3, min_periods=1).mean()
        
        # Forward fill then backward fill for trend features
        trend_cols = [col for col in df.columns if '_trend_' in col or '_3h_mean' in col]
        for col in trend_cols:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    def _select_core_features(self, df: pd.DataFrame) -> List[str]:
        """Select core features to prevent overfitting"""
        
        core_features = [
            # Basic weather parameters
            'temperature_2m', 'relative_humidity_2m', 'pressure_msl', 
            'wind_speed_10m', 'wind_gusts_10m', 'dew_point_2m',
            'precipitation_probability', 'precipitation', 'cloud_cover',
            'current_weather_severity', 'weather_code',
            
            # Time features
            'hour', 'month', 'season', 'is_daytime',
            
            # Derived features
            'temp_dewpoint_spread', 'wind_gust_ratio', 'instability_indicator',
            
            # Short trends only
            'pressure_trend_1h', 'temp_trend_1h', 'humidity_trend_1h',
            
            # Smoothed values
            'pressure_msl_3h_mean', 'temperature_2m_3h_mean', 'wind_speed_10m_3h_mean'
        ]
        
        # Only include features that exist in the dataframe
        available_features = [f for f in core_features if f in df.columns]
        
        print(f"Selected {len(available_features)} core features")
        return available_features
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Better missing value handling"""
        
        # For trend features, fill with 0 (no change)
        trend_features = [col for col in X.columns if 'trend' in col]
        for col in trend_features:
            X[col] = X[col].fillna(0)
        
        # For other features, use median
        for col in X.columns:
            if col not in trend_features:
                X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42):
        """Train improved XGBoost model with regularization"""
        
        # Use time-based split for weather data (more realistic)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Time-based split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Feature selection to reduce overfitting
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(15, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature names
        selected_features = [self.feature_columns[i] for i in self.feature_selector.get_support(indices=True)]
        print(f"Selected {len(selected_features)} most important features")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # XGBoost with strong regularization to prevent overfitting
        self.model = xgb.XGBRegressor(
            # Reduce complexity
            n_estimators=100,  # Reduced from 200
            max_depth=4,       # Reduced from 6
            learning_rate=0.05, # Reduced from 0.1
            
            # Add regularization
            reg_alpha=1.0,     # L1 regularization
            reg_lambda=1.0,    # L2 regularization
            gamma=0.1,         # Minimum split loss
            
            # Prevent overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5, # Increased from 1
            
            # Other settings
            random_state=random_state,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        
        print("Training regularized XGBoost model...")
        
        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Clip to valid range
        y_train_pred = np.clip(y_train_pred, 0, 100)
        y_test_pred = np.clip(y_test_pred, 0, 100)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation for more robust evaluation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_root_mean_squared_error'
        )
        cv_rmse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store feature importance
        self.feature_importance_ = dict(zip(selected_features, self.model.feature_importances_))
        
        print(f"\n=== IMPROVED MODEL PERFORMANCE ===")
        print(f"Train RMSE: {train_rmse:.3f}")
        print(f"Test RMSE:  {test_rmse:.3f}")
        print(f"CV RMSE:    {cv_rmse:.3f} (+/- {cv_std:.3f})")
        print(f"Train R²:   {train_r2:.3f}")
        print(f"Test R²:    {test_r2:.3f}")
        print(f"Overfitting Gap: {train_rmse - test_rmse:.3f}")
        
        if abs(train_rmse - test_rmse) < 2.0:
            print("✅ Good generalization - low overfitting")
        else:
            print("⚠️  Still some overfitting - consider more regularization")
        
        # Feature importance
        print(f"\n=== TOP 10 IMPORTANT FEATURES ===")
        sorted_features = sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<25} {importance:.4f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'overfitting_gap': train_rmse - test_rmse,
            'selected_features': selected_features
        }
    
    def predict(self, weather_data: Dict) -> float:
        """Predict next3h_max_weathercode for new weather data"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert to DataFrame
        df_input = pd.DataFrame([weather_data])
        
        # Add time features
        if 'time' in weather_data:
            df_input['time'] = pd.to_datetime(weather_data['time'])
        else:
            df_input['time'] = pd.Timestamp.now()
        
        df_input['hour'] = df_input['time'].dt.hour
        df_input['day_of_year'] = df_input['time'].dt.dayofyear
        df_input['month'] = df_input['time'].dt.month
        df_input['is_daytime'] = ((df_input['hour'] >= 6) & (df_input['hour'] <= 18)).astype(int)
        df_input['season'] = (df_input['month'] % 12 // 3).astype(int)
        
        # Create derived features
        df_input['temp_dewpoint_spread'] = weather_data.get('temperature_2m', 25) - weather_data.get('dew_point_2m', 20)
        df_input['temp_apparent_diff'] = weather_data.get('temperature_2m', 25) - weather_data.get('apparent_temperature', 25)
        df_input['pressure_surface_diff'] = weather_data.get('pressure_msl', 1013) - weather_data.get('surface_pressure', 1013)
        df_input['wind_gust_ratio'] = weather_data.get('wind_gusts_10m', 0) / (weather_data.get('wind_speed_10m', 1) + 1)
        df_input['instability_indicator'] = int(
            (df_input['temp_dewpoint_spread'].iloc[0] < 5) and 
            (weather_data.get('relative_humidity_2m', 50) > 75)
        )
        
        # Weather severity
        weather_severity_map = {
            0: 0, 1: 5, 2: 10, 3: 15, 45: 25, 48: 30,
            51: 20, 53: 25, 55: 30, 56: 35, 57: 40,
            61: 30, 63: 40, 65: 50, 66: 55, 67: 60,
            71: 35, 73: 45, 75: 55, 77: 50,
            80: 40, 81: 50, 82: 60, 85: 55, 86: 65,
            95: 75, 96: 85, 99: 95
        }
        df_input['current_weather_severity'] = weather_severity_map.get(weather_data.get('weather_code', 0), 15)
        
        # Trend features (set to 0 for single prediction)
        df_input['pressure_trend_1h'] = 0
        df_input['temp_trend_1h'] = 0
        df_input['humidity_trend_1h'] = 0
        
        # 3h means (use current values)
        df_input['pressure_msl_3h_mean'] = weather_data.get('pressure_msl', 1013)
        df_input['temperature_2m_3h_mean'] = weather_data.get('temperature_2m', 25)
        df_input['wind_speed_10m_3h_mean'] = weather_data.get('wind_speed_10m', 10)
        
        # Ensure all features exist
        for feature in self.feature_columns:
            if feature not in df_input.columns:
                if feature in weather_data:
                    df_input[feature] = weather_data[feature]
                else:
                    df_input[feature] = 0
        
        # Select features in correct order
        X_pred = df_input[self.feature_columns]
        
        # Apply feature selection
        X_pred_selected = self.feature_selector.transform(X_pred)
        
        # Scale
        X_pred_scaled = self.scaler.transform(X_pred_selected)
        
        # Predict
        prediction = self.model.predict(X_pred_scaled)[0]
        prediction = float(np.clip(prediction, 0, 100))
        
        return prediction
    
    def save_model(self, filepath: str = 'improved_weather_predictor.joblib'):
        """Save model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'feature_importance': self.feature_importance_
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'improved_weather_predictor.joblib'):
        """Load model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.feature_importance_ = model_data.get('feature_importance', {})
        print(f"Model loaded from {filepath}")


# Training function
def train_improved_model():
    """Train the improved model"""
    predictor = ImprovedWeatherCodePredictor()
    
    # Load data
    csv_file = 'amd_with_next3h_max_weathercode.csv'
    X, y = predictor.load_and_prepare_data(csv_file)
    
    # Train with regularization
    results = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model()
    
    return predictor, results


# Backend prediction function
def predict_weather_code(weather_data: Dict, model_path: str = 'improved_weather_predictor.joblib') -> float:
    """Backend prediction function"""
    predictor = ImprovedWeatherCodePredictor()
    predictor.load_model(model_path)
    return predictor.predict(weather_data)


# Main execution
if __name__ == "__main__":
    
    print("=== TRAINING IMPROVED MODEL ===")
    predictor, results = train_improved_model()
    
    print("\n=== TESTING PREDICTIONS ===")
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
    
    prediction = predictor.predict(sample_data)
    print(f"Improved prediction: {prediction:.1f}")
