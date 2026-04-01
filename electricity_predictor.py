import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from datetime import datetime, date, timedelta
import requests
import json
import os
import re

class ElectricityUsagePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.baselines = {}
        self.building_columns = []
        self.training_features = {}
        self.recommendations_df = None
        
        # Pure Isolation Forest approach
        self.pattern_anomaly_detectors = {}
        self.anomaly_contamination = 0.1
        
    def load_recommendations(self, recommendations_file="recommendations.csv"):
        """
        Load existing energy recommendations dataset
        """
        try:
            if os.path.exists(recommendations_file):
                self.recommendations_df = pd.read_csv(recommendations_file)
                print(f" Loading recommendations from: {recommendations_file}")
                self.recommendations_df = pd.read_csv(recommendations_file)
                print(f" Loaded recommendations for {len(self.recommendations_df)} buildings")
                print(f" Recommendations columns: {list(self.recommendations_df.columns)}")
                for i, col in enumerate(self.recommendations_df.columns):
                     print(f"   Column {i}: '{col}'")
                # Print first few building names to help with debugging
                if len(self.recommendations_df) > 0:
                    building_col = self.recommendations_df.columns[0]
                    print(f" Sample building names in recommendations:")
                    for i, name in enumerate(self.recommendations_df[building_col].head(5)):
                        print(f"   {i+1}. {name}")
                    if len(self.recommendations_df.columns) > 1:
                           rec_col = self.recommendations_df.columns[1]
                           print(f"\n Sample recommendation text from column '{rec_col}':")
                           for i in range(min(3, len(self.recommendations_df))):
                                rec_text = self.recommendations_df.iloc[i][rec_col]
                                print(f"   Building '{self.recommendations_df.iloc[i][building_col]}':")
                                print(f"      '{str(rec_text)[:100]}...'")    
                print(f"\n🔍 Checking for null values:")
                for col in self.recommendations_df.columns:
                    null_count = self.recommendations_df[col].isnull().sum()
                    print(f"   '{col}': {null_count} null values")                
                return True
            else:
                print(f" Recommendations file {recommendations_file} not found")
                return False
        except Exception as e:
            print(f" Error loading recommendations: {e}")
            import traceback
            traceback.print_exc()
            return False

    """def extract_building_number(self, building_name):
        
        # Try different patterns to extract number
        patterns = [
            r'Elec_?(\d+)',  # Matches Elec_1, Elec1, etc.
            r'Building_?(\d+)',  # Matches Building_1, Building1, etc.
            r'Bldg_?(\d+)',  # Matches Bldg_1, Bldg1, etc.
            r'(\d+)',  # Just numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, building_name)
            if match:
                return match.group(1)
        return None"""

    def get_recommendations(self, building_name, anomaly_level):
        
        #building_number = self.extract_building_number(building_name)
        
        building_match = None
        building_col = self.recommendations_df.columns[0] 
        
        
        for idx, row in self.recommendations_df.iterrows():
            rec_building_name = str(row[building_col])
            
            
           
            
           
            
            # Strategy 3: Check for substring match
            if (building_name.lower() in rec_building_name.lower() or 
                rec_building_name.lower() in building_name.lower()):
                building_match = row
                break
        
        if building_match is None:
            print(f"⚠️  No match found for {building_name}, using fallback recommendations")
            print(f"   Available buildings in recommendations: {list(self.recommendations_df[building_col].head(10))}")
            return self.get_fallback_recommendations(building_name, anomaly_level)
        
        recommendations = []
        
        
        
        if anomaly_level == 'low':
            # Extremely low usage - possible equipment failure
            low_cols = [col for col in self.recommendations_df.columns 
                       if any(keyword in col for keyword in ['Recommendation 1'])]
            if low_cols:
                for col in low_cols[:1]:  # Take only 1 recommendation for low
                    if col in building_match and pd.notna(building_match[col]):
                        rec_text = str(building_match[col]).strip()
                        if rec_text and rec_text.lower() != 'nan':
                            recommendations.append(rec_text)
        elif anomaly_level == 'moderate':
            # Moderate wasteful usage
            pattern_cols = [col for col in self.recommendations_df.columns 
                          if 'Recommendation 2' in col]
            if pattern_cols:
                for col in pattern_cols[:1]:  # Take 2 recommendations
                    if col in building_match and pd.notna(building_match[col]):
                        rec_text = str(building_match[col]).strip()
                        if rec_text and rec_text.lower() != 'nan':
                            recommendations.append(rec_text)
        elif anomaly_level == 'high':
            # High wasteful usage
            high_cols = [col for col in self.recommendations_df.columns 
                        if any(keyword in col for keyword in ['Recommendation 3'])]
            if high_cols:
                for col in high_cols[:1]:  # Take 3 recommendations
                    if col in building_match and pd.notna(building_match[col]):
                        rec_text = str(building_match[col]).strip()
                        if rec_text and rec_text.lower() != 'nan':
                            recommendations.append(rec_text)
        
        print(f"📋 Found {len(recommendations)} recommendations for {building_name}")
        
        # If we didn't find enough recommendations, add fallback ones
        if len(recommendations) == 0:
            print(f"⚠️  No valid recommendations found in dataset for {building_name}, using fallback")
            return self.get_fallback_recommendations(building_name, anomaly_level)
        
        return recommendations
    
   

    def get_savings_feedback(self, building_name, error_percentage,anomaly_level):
        """if  anomaly_level in ['high', 'moderate', 'low']:
            return []"""
        
          # Significant savings
        if error_percentage < -50:
                return [f"{building_name}  Check for system errors (usage is too low)"]
               
        if error_percentage < 0:
                return [f"{building_name}is achieving energy savings."]
               
            # For normal operation (no savings, but not wasteful)
        return [f"{building_name} operating within expected energy range."]

    def train_pattern_anomaly_detectors(self, df):
        
        print("\n TRAINING PATTERN ANOMALY DETECTORS...")
        
        trained_count = 0
        
        for building in df['building'].unique():
            print(f"  Training pattern detector for {building}...")
            building_data = df[df['building'] == building]
            
            if len(building_data) < 25:
                print(f"    Insufficient data ({len(building_data)} samples)")
                continue
            
            feature_vectors = []
            
            # Create feature vectors from raw data
            for idx, row in building_data.iterrows():
                feature_vector = [
                    row['usage'],           # Actual electricity usage
                    row['temperature'],     # Temperature
                    row['humidity'],        # Humidity
                    row['day_of_week'],     # Day of week (0-6)
                    row['month'],           # Month (1-12)
                    row['is_weekend']       # Weekend flag (0/1)
                ]
                feature_vectors.append(feature_vector)
            
            if len(feature_vectors) >= 20:
                X = np.array(feature_vectors)
                
                # Train Isolation Forest
                iso_forest = IsolationForest(
                    contamination=self.anomaly_contamination,
                    random_state=42,
                    n_estimators=100,
                    max_samples='auto'
                )
                
                iso_forest.fit(X)
                self.pattern_anomaly_detectors[building] = iso_forest
                
                # Test the detector on training data
                predictions = iso_forest.predict(X)
                anomaly_count = np.sum(predictions == -1)
                anomaly_rate = (anomaly_count / len(predictions)) * 100
                
                print(f"   Trained - {len(feature_vectors)} samples, {anomaly_rate:.1f}% anomalies in training")
                trained_count += 1
        
        print(f" Trained pattern anomaly detectors for {trained_count} buildings")
    
    def detect_anomaly_combined(self, building_name, actual_usage, weather_data, prediction_error_percentage, baseline_info):
       
       
        iso_result = 'unknown'
        if building_name in self.pattern_anomaly_detectors:
            iso_result = self.detect_anomaly_isolation_forest(building_name, actual_usage, weather_data)
        
        # SMART LOGIC: Reward savings, penalize waste
        if prediction_error_percentage > 50:  # >50% above prediction = HIGH WASTE
            return 'high'
        elif prediction_error_percentage > 20:  # 30-50% above prediction = MODERATE WASTE
            return 'moderate'
        elif prediction_error_percentage >5:
            return 'low'
        elif prediction_error_percentage <=0:
            return 'normal'
        else:
            iso_result = self.detect_anomaly_isolation_forest(building_name, actual_usage, weather_data)
            return iso_result
    """else: 
           if iso_result in ['high', 'moderate', 'low']:
                return iso_result
           return 'normal'"""
    
    def detect_anomaly_isolation_forest(self, building_name, actual_usage, weather_data):
        """
        Isolation Forest detection
        """
        if building_name not in self.pattern_anomaly_detectors:
            return 'unknown'
        
        # Get current date features
        today = date.today()
        current_date = pd.Timestamp(today)
        
        # Create feature vector from current conditions
        feature_vector = [
            actual_usage,                           # Today's actual usage
            weather_data['temperature'],            # Current temperature
            weather_data['humidity'],               # Current humidity
            current_date.dayofweek,                 # Day of week (0-6)
            current_date.month,                     # Current month
            1 if current_date.dayofweek >= 5 else 0 # Weekend flag
        ]
        
        try:
            # Detect anomaly
            X = np.array([feature_vector])
            prediction = self.pattern_anomaly_detectors[building_name].predict(X)[0]
            
            if prediction == -1:
                # It's anomalous - get anomaly score for severity
                anomaly_score = self.pattern_anomaly_detectors[building_name].decision_function(X)[0]
                
                # Convert score to anomaly level
                if anomaly_score < -0.3:    # Very anomalous
                    return 'high'
                elif anomaly_score < -0.1:  # Moderately anomalous
                    return 'moderate'
                else:                       # Slightly anomalous
                    return 'low'
            else:
                return 'normal'
                
        except Exception as e:
            print(f" Isolation Forest error: {e}")
            return 'unknown'
    
    def get_most_appropriate_baseline(self, baseline_info):
       
        if not baseline_info:
            return 0
        
        # 1. Try temperature-specific baseline
        if baseline_info.get('matched_temperature') and baseline_info.get('temperature_baseline'):
            temp_baseline = baseline_info['temperature_baseline']['mean']
            if temp_baseline > 0 and baseline_info['temperature_baseline']['count'] >= 5:
                return temp_baseline
        
        # 2. Try season-specific baseline
        if baseline_info.get('matched_season') and baseline_info.get('season_baseline'):
            season_baseline = baseline_info['season_baseline']['mean']
            if season_baseline > 0 and baseline_info['season_baseline']['count'] >= 10:
                return season_baseline
        
        # 3. Try day-type baseline
        if baseline_info.get('day_type_baseline'):
            day_type_baseline = baseline_info['day_type_baseline']['mean']
            if day_type_baseline > 0 and baseline_info['day_type_baseline']['count'] >= 10:
                return day_type_baseline
        
        # 4. Fall back to overall baseline
        if baseline_info.get('overall'):
            overall_baseline = baseline_info['overall']['mean']
            if overall_baseline > 0:
                return overall_baseline
        
        return 0

    def establish_weather_aware_baselines(self, df):
        """
        Establish weather-aware baselines for each building
        """
        print("\n ESTABLISHING WEATHER-AWARE BASELINES...")
        baselines = {}
        
        if 'building' not in df.columns:
            print("❌ No building column in data for baselines")
            return baselines
        
        # Define temperature ranges for baseline calculation
        temp_ranges = {
            'cold': (-10, 10),
            'mild': (10, 20),  
            'warm': (20, 30),
            'hot': (30, 50)
        }
        
        for building in df['building'].unique():
            building_data = df[df['building'] == building]
            
            if len(building_data) == 0:
                continue
            
            building_baselines = {
                'overall': self._calculate_stats(building_data['usage']),
                'by_day_type': {},
                'by_temperature': {},
                'by_season': {}
            }
            
            # Day Type Baselines
            weekdays_data = building_data[building_data['day_of_week'] < 5]
            weekends_data = building_data[building_data['day_of_week'] >= 5]
            
            building_baselines['by_day_type'] = {
                'weekdays': self._calculate_stats(weekdays_data['usage']),
                'weekends': self._calculate_stats(weekends_data['usage'])
            }
            
            # Temperature Baselines
            for temp_name, (temp_low, temp_high) in temp_ranges.items():
                temp_data = building_data[
                    (building_data['temperature'] >= temp_low) & 
                    (building_data['temperature'] < temp_high)
                ]
                building_baselines['by_temperature'][temp_name] = self._calculate_stats(temp_data['usage'])
            
            # Seasonal Baselines
            seasons = {
                'winter': [12, 1, 2],
                'spring': [3, 4, 5], 
                'summer': [6, 7, 8],
                'fall': [9, 10, 11]
            }
            
            for season_name, months in seasons.items():
                season_data = building_data[building_data['month'].isin(months)]
                building_baselines['by_season'][season_name] = self._calculate_stats(season_data['usage'])
            
            baselines[building] = building_baselines
            
            # Print baseline summary
            self._print_baseline_summary(building, building_baselines)
        
        self.baselines = baselines
        print(f"\n Established weather-aware baselines for {len(baselines)} buildings")
        return baselines
    
    def _calculate_stats(self, data):
        """Calculate statistics for a dataset"""
        if len(data) == 0:
            return {'mean': 0, 'median': 0, 'std_dev': 0, 'count': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std_dev': data.std(),
            'count': len(data),
            'min': data.min(),
            'max': data.max()
        }
    
    def _print_baseline_summary(self, building_name, baselines):
        """Print a summary of baselines for a building"""
        overall = baselines['overall']
        print(f"   {building_name}: {overall['mean']:.1f} kWh ({overall['count']} days)")

    def get_current_baseline(self, building_name, current_weather, current_date=None):
        """
        Get the most appropriate baseline based on CURRENT weather and date
        """
        if building_name not in self.baselines:
            print(f"  No baselines for {building_name}")
            return None
        
        if current_date is None:
            current_date = date.today()
        
        current_date = pd.Timestamp(current_date)
        is_weekend = current_date.dayofweek >= 5
        current_month = current_date.month
        current_temp = current_weather['temperature']
        
        building_baselines = self.baselines[building_name]
        
        baseline_info = {
            'overall': building_baselines['overall'],
            'day_type': 'weekends' if is_weekend else 'weekdays',
            'day_type_baseline': building_baselines['by_day_type']['weekends' if is_weekend else 'weekdays'],
            'matched_temperature': None,
            'matched_season': None
        }
        
        # Find matching temperature range
        temp_ranges = {
            'cold': (-10, 10),
            'mild': (10, 20), 
            'warm': (20, 30),
            'hot': (30, 50)
        }
        
        for temp_name, (temp_low, temp_high) in temp_ranges.items():
            if temp_low <= current_temp < temp_high:
                baseline_info['matched_temperature'] = temp_name
                baseline_info['temperature_baseline'] = building_baselines['by_temperature'][temp_name]
                break
        
        # Find matching season
        seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8], 
            'fall': [9, 10, 11]
        }
        
        for season_name, months in seasons.items():
            if current_month in months:
                baseline_info['matched_season'] = season_name
                baseline_info['season_baseline'] = building_baselines['by_season'][season_name]
                break
        
        return baseline_info

    def get_current_weather(self, city="Vancouver"):
        """
        Get current weather data from OpenWeatherMap API
        """
        api_key = "b1b15e88fa797225412429c1c50c122a1"
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description']
                }
                print(f"  Current weather in {city}: {weather_data['temperature']}°C, {weather_data['humidity']}% humidity")
                return weather_data
            else:
                print(f"  Weather API error, using sample data")
                return self.get_sample_weather_data()
                
        except Exception as e:
            print(f"  Error fetching weather data, using sample data: {e}")
            return self.get_sample_weather_data()
    
    def get_sample_weather_data(self):
        """
        Provide sample weather data when API is not available
        """
        print(" Using sample weather data for prediction")
        current_month = datetime.now().month
        if current_month in [12, 1, 2]:
            return {'temperature': 5.5, 'humidity': 75, 'description': 'cloudy'}
        elif current_month in [3, 4, 5]:
            return {'temperature': 15.5, 'humidity': 65, 'description': 'partly cloudy'}
        elif current_month in [6, 7, 8]:
            return {'temperature': 25.5, 'humidity': 60, 'description': 'clear sky'}
        else:
            return {'temperature': 18.5, 'humidity': 70, 'description': 'light rain'}

    def prepare_data_simple(self, electricity_df, weather_df):
       
        
        
        # Clean electricity data
        electricity_df = electricity_df.drop(columns=['Unnamed: 26'], errors='ignore')
        
        # Get building columns
        self.building_columns = [col for col in electricity_df.columns 
                               if col != 'Date' and 'Elec' in col]
        print(f" Found {len(self.building_columns)} building columns")
        
        # Convert dates
        electricity_df = electricity_df.copy()
        electricity_df['date'] = pd.to_datetime(electricity_df['Date'], dayfirst=True, errors='coerce')
        electricity_df = electricity_df.dropna(subset=['date'])
        
        weather_df = weather_df.copy()
        weather_df['date'] = pd.to_datetime(weather_df['Date'], dayfirst=True, errors='coerce')
        weather_df = weather_df.dropna(subset=['date'])
        
        # Create combined dataset
        combined_data = []
        
        all_dates = sorted(set(electricity_df['date']).intersection(set(weather_df['date'])))
        print(f" Common dates: {len(all_dates)}")
        
        for current_date in all_dates:
            elec_data = electricity_df[electricity_df['date'] == current_date]
            weather_data = weather_df[weather_df['date'] == current_date]
            
            if not elec_data.empty and not weather_data.empty:
                for building in self.building_columns:
                    if building in elec_data.columns:
                        usage = elec_data[building].iloc[0]
                        if not pd.isna(usage):
                            row = {
                                'date': current_date,
                                'building': building,
                                'usage': usage,
                                'temperature': weather_data['Temperature (°C)'].iloc[0],
                                'humidity': weather_data['Humidity (%)'].iloc[0]
                            }
                            combined_data.append(row)
        
        combined_df = pd.DataFrame(combined_data)
        print(f" Combined data shape: {combined_df.shape}")
        
        if len(combined_df) == 0:
            print(" No combined data created!")
            return None
        
        # Create features
        combined_df = self.create_features(combined_df)
        
        return combined_df
    
    def create_features(self, df):
        
        df = df.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        if len(df) == 0:
            return df
        
        # Create basic time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        df['season'] = (df['month'] % 12 + 3) // 3
        
        return df
    
    def prepare_building_features(self, df, target_building):
        
        building_data = df[df['building'] == target_building].copy()
        
        if len(building_data) == 0:
            print(f" No data for {target_building}")
            return pd.DataFrame(), pd.Series(), []
        
        features = [
            'day_of_week', 'month', 'is_weekend', 
            'season', 'temperature', 'humidity'
        ]
        
        available_features = [f for f in features if f in building_data.columns]
        
        self.training_features[target_building] = available_features
        
        X = building_data[available_features]
        y = building_data['usage']
        
        print(f" Features for {target_building}: {len(available_features)} features, {len(X)} samples")
        
        return X, y, available_features
    
    def train_models(self, df):
       
        trained_count = 0
        
        if 'building' not in df.columns:
            print(" No building column for training")
            return
        
        buildings_with_data = df['building'].unique()
        print(f" Buildings with data: {len(buildings_with_data)}")
        
        for building in buildings_with_data:
            print(f" Training {building}...")
            
            building_data = df[df['building'] == building]
            if len(building_data) < 10:
                print(f"    Insufficient data ({len(building_data)} samples)")
                continue
            
            X, y, feature_names = self.prepare_building_features(df, building)
            
            if len(X) < 10:
                print(f"    Insufficient features after preparation")
                continue
            
            # Clean data
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 10:
                
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale and train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Store model
            self.models[building] = model
            self.scalers[building] = scaler
            self.feature_importances[building] = dict(zip(feature_names, model.feature_importances_))
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"   MAE: {mae:.2f} kWh, Samples: {len(X_train)}")
            trained_count += 1
        
        print(f" {trained_count}/{len(buildings_with_data)} models trained successfully")
        
        # Train Isolation Forest detectors
        if trained_count > 0:
            self.train_pattern_anomaly_detectors(df)
            print(f" Pure Isolation Forest detection ready for {len(self.pattern_anomaly_detectors)} buildings")
    
    def prepare_prediction_features(self, weather_data):
       
        today = date.today()
        
        prediction_date = pd.Timestamp(today)
        day_of_week = prediction_date.dayofweek
        month = prediction_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        season = (month % 12 + 3) // 3
        
        features = {
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'season': season,
            'temperature': weather_data['temperature'],
            'humidity': weather_data['humidity']
        }
        
        return features
    
    def predict_today_usage(self, building_name, actual_usage=None):
        """
        Predict today's electricity usage with SMART anomaly detection
        """
        if building_name not in self.models:
            print(f"❌ No model for {building_name}")
            return None
        
        print("  Getting weather...")
        weather_data = self.get_current_weather()
        
        # Prepare prediction features
        feature_dict = self.prepare_prediction_features(weather_data)
        
        if building_name not in self.training_features:
            print(f" No feature information for {building_name}")
            return None
        
        training_features = self.training_features[building_name]
        
        
        feature_array = []
        for feature in training_features:
            if feature in feature_dict:
                feature_array.append(feature_dict[feature])
            else:
                feature_array.append(0)
        
        X = np.array([feature_array])
        
        
        scaler = self.scalers[building_name]
        X_scaled = scaler.transform(X)
        prediction = self.models[building_name].predict(X_scaled)[0]
        
        print(f" PREDICTION: {prediction:.2f} kWh")
        
        # Get current baseline
        current_baseline = self.get_current_baseline(building_name, weather_data)
        
        result = {
            'building': building_name,
            'date': date.today(),
            'predicted_usage_kwh': float(prediction),
            'weather_temperature': weather_data['temperature'],
            'weather_humidity': weather_data['humidity'],
            'weather_description': weather_data['description']
        }
        
        # Add baseline information
        if current_baseline:
            result['baseline_info'] = current_baseline
            result['day_type'] = current_baseline['day_type']
            result['overall_baseline'] = current_baseline['overall']['mean']
            
            # Add specific baseline values
            if 'temperature_baseline' in current_baseline:
                result['temperature_baseline'] = current_baseline['temperature_baseline']['mean']
            if 'season_baseline' in current_baseline:
                result['season_baseline'] = current_baseline['season_baseline']['mean']
            if 'day_type_baseline' in current_baseline:
                result['day_type_baseline'] = current_baseline['day_type_baseline']['mean']
        
        # If actual usage was provided, use SMART anomaly detection
        if actual_usage is not None:
            result['actual_usage_kwh'] = float(actual_usage)
            
            # Calculate prediction error and percentage
            prediction_error = actual_usage - prediction
            prediction_error_percentage = (prediction_error / prediction) * 100 if prediction != 0 else 0
            
            result['prediction_error_kwh'] = float(prediction_error)
            result['prediction_error_percentage'] = float(prediction_error_percentage)
            
            # SMART anomaly detection - focus on waste, reward savings
            anomaly_level = self.detect_anomaly_combined(
                building_name, 
                actual_usage, 
                weather_data, 
                prediction_error_percentage,
                current_baseline
            )
            
            result['anomaly_level'] = anomaly_level
            result['detection_method'] = 'smart_combined'
            
            
            if anomaly_level in ['high', 'moderate', 'low']:
                
                result['recommendations'] = self.get_recommendations(building_name, anomaly_level)
                result['feedback_type'] = 'recommendation'
            else:
                
                result['recommendations'] = self.get_savings_feedback(building_name, prediction_error_percentage,anomaly_level)
                result['feedback_type'] = 'praise'
            
            print(f" ANOMALY DETECTION: {anomaly_level.upper()}")
            print(f" Prediction Error: {prediction_error:+.1f} kWh ({prediction_error_percentage:+.1f}%)")
            print(f"  Feedback type: {result['feedback_type']}")
        
        return result


# MAIN METHOD
def main():
    ELECTRICITY_FILE = "new dataset.csv"
    WEATHER_FILE = "weather.csv"
    RECOMMENDATIONS_FILE = "energy_recommendations.csv"
    
    if not os.path.exists(ELECTRICITY_FILE):
        print(f" {ELECTRICITY_FILE} not found!")
        return
    
    if not os.path.exists(WEATHER_FILE):
        print(f" {WEATHER_FILE} not found!")
        return
    
    try:
        # Load data
        print(" Loading datasets...")
        electricity_data = pd.read_csv(ELECTRICITY_FILE)
        weather_data = pd.read_csv(WEATHER_FILE)
        
        print(" Data loaded!")
        
        # Initialize predictor
        predictor = ElectricityUsagePredictor()
        if not predictor.load_recommendations(RECOMMENDATIONS_FILE):
            print("  Continuing without recommendations dataset")
        
        # Process data
        combined_data = predictor.prepare_data_simple(electricity_data, weather_data)
        
        if combined_data is None:
            print(" No data processed successfully!")
            return
        
        print(f" Final combined data: {combined_data.shape}")
        
        # Train models
        predictor.establish_weather_aware_baselines(combined_data)
        predictor.train_models(combined_data)
        
        if not predictor.models:
            print(" No models were trained successfully!")
            return
        
        # User interaction
        print("\n" + "="*60)
        print("🔌 TODAY'S ELECTRICITY USAGE INPUT")
        print("="*60)
        
        available_models = list(predictor.models.keys())
        print("Available buildings:")
        for i, building in enumerate(available_models, 1):
            print(f"{i}. {building}")
        
        try:
            choice = input(f"\nSelect building (1-{len(available_models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                building = available_models[idx]
                
                actual_usage = float(input(f"\nEnter TODAY'S ACTUAL usage for {building} (kWh): ").strip())
                
                if actual_usage < 0:
                    print(" Usage cannot be negative")
                    return
                
                # Get prediction and anomaly detection
                print(f"\n Analyzing {building}...")
                result = predictor.predict_today_usage(building, actual_usage)
                
                if result is None:
                    print(" Failed to get prediction. Please check if the building has a trained model.")
                    return
                
                if 'actual_usage_kwh' in result:
                    print(f"\n" + "="*60)
                    print(f" ANALYSIS RESULTS FOR {result['building']}")
                    print("="*60)
                    print(f" Date: {result['date']}")
                    print(f"  Weather: {result['weather_temperature']}°C, {result['weather_humidity']}% humidity")
                    print(f" Predicted Usage: {result['predicted_usage_kwh']:.2f} kWh")
                    print(f" Actual Usage: {result['actual_usage_kwh']:.2f} kWh")
                    print(f"  Prediction Error: {result['prediction_error_kwh']:+.2f} kWh ({result['prediction_error_percentage']:+.2f}%)")
                    print(f" Anomaly Level: {result['anomaly_level'].upper()}")
                    print(f" Detection Method: {result['detection_method']}")
                    print("="*60)
                    
                    # Check if recommendations exists and is not None
                    if 'recommendations' in result and result['recommendations'] is not None:
                        if result['feedback_type'] == 'recommendation':
                            print(f"\n  RECOMMENDATIONS ({len(result['recommendations'])}):")
                            print("-" * 40)
                        else:
                            print(f"\n FEEDBACK:")
                            print("-" * 40)
                        
                        for i, rec in enumerate(result['recommendations'], 1):
                            print(f"{i}. {rec}")
                    else:
                        print(f"\n FEEDBACK: Operating normally")
                    print("="*60)
                else:
                    print(" Error getting prediction")
                    
            else:
                print(" Invalid selection")
                
        except (ValueError, IndexError):
            print(" Please enter a valid number")
                
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()