from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from datetime import datetime, date, timedelta
import requests
import json
import os
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
warnings.filterwarnings('ignore')

# Import your existing class
from electricity_predictor import ElectricityUsagePredictor

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production

# Global variable to store the predictor instance
predictor = None
combined_data = None

# Building name mapping - This maps electricity dataset building names to recommendation dataset building names


def initialize_predictor():
    """Initialize the predictor with data"""
    global predictor, combined_data
    
    try:
        predictor = ElectricityUsagePredictor()
        
        # Load data
        ELECTRICITY_FILE = "new dataset.csv"
        WEATHER_FILE = "weather.csv"
        RECOMMENDATIONS_FILE = "recommendations.csv"
        
        if not os.path.exists(ELECTRICITY_FILE):
            return False, f"Electricity file {ELECTRICITY_FILE} not found"
        if not os.path.exists(WEATHER_FILE):
            return False, f"Weather file {WEATHER_FILE} not found"
        
        electricity_data = pd.read_csv(ELECTRICITY_FILE)
        weather_data = pd.read_csv(WEATHER_FILE)
        
        # Load recommendations
        if not os.path.exists(RECOMMENDATIONS_FILE):
            print(f"⚠️  Recommendations file {RECOMMENDATIONS_FILE} not found, will use fallback recommendations")
        else:
            predictor.load_recommendations(RECOMMENDATIONS_FILE)
            print(f"✅ Recommendations loaded successfully")
            # Print available buildings in recommendations
            if predictor.recommendations_df is not None and len(predictor.recommendations_df) > 0:
                building_col = predictor.recommendations_df.columns[0]
                print(f"📋 Buildings in recommendations file:")
                for i, name in enumerate(predictor.recommendations_df[building_col].head(10)):
                    print(f"   {i+1}. {name}")
        
        # Process data
        combined_data = predictor.prepare_data_simple(electricity_data, weather_data)
        if combined_data is None or len(combined_data) == 0:
            return False, "No data processed successfully"
        
        # Train models
        predictor.establish_weather_aware_baselines(combined_data)
        predictor.train_models(combined_data)
        
        if not predictor.models:
            return False, "No models were trained successfully"
        
        print(f"✅ Predictor initialized with {len(predictor.models)} building models")
        return True, "Predictor initialized successfully"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error initializing predictor: {str(e)}"



@app.route('/')
def index():
    """Main page"""
    global predictor
    
    if predictor is None:
        success, message = initialize_predictor()
        if not success:
            return render_template('error.html', error_message=message)
    
    buildings = list(predictor.models.keys()) if predictor and predictor.models else []
    
    # Create display names for buildings
    building_display_names = []
    
    
    return render_template('index.html', buildings=buildings, building_display_names=building_display_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    global predictor
    
    try:
        building_name = request.form['building']
        actual_usage = float(request.form['actual_usage'])
        
        if predictor is None:
            return jsonify({'error': 'Predictor not initialized'})
        
        if building_name not in predictor.models:
            return jsonify({'error': f'No model found for {building_name}'})
        
        # Get recommendation building name for debugging
       
        
        
        # Get prediction
        result = predictor.predict_today_usage(building_name, actual_usage)
        
        if not result or 'actual_usage_kwh' not in result:
            return jsonify({'error': 'Error getting prediction'})
        
        # Check if we got recommendations from dataset or fallback
        recommendations = result.get('recommendations', [])
        if recommendations:
            print(f" Got {len(recommendations)} recommendations")
            for i, rec in enumerate(recommendations):
                print(f"   {i+1}. {rec}")
        else:
            print(f"  No recommendations found for {building_name}")
        
        # Prepare response data
        response_data = {
            'success': True,
            'building': building_name,
            
            'date': str(result['date']),
            'predicted_usage': round(result['predicted_usage_kwh'], 2),
            'actual_usage': round(result['actual_usage_kwh'], 2),
            'prediction_error': round(result.get('prediction_error_kwh', 0), 2),
            'error_percentage': round(result.get('prediction_error_percentage', 0), 2),
            'anomaly_level': result.get('anomaly_level', 'unknown'),
            'feedback_type': result.get('feedback_type', 'normal'),
            'weather_temperature': result['weather_temperature'],
            'weather_humidity': result['weather_humidity'],
            'weather_description': result['weather_description'],
            'recommendations': recommendations,
            'recommendation_count': len(recommendations),
            'detection_method': result.get('detection_method', 'unknown'),
            'recommendations_source': 'dataset' if recommendations and any('dataset' in str(r).lower() for r in recommendations) else 'fallback'
        }
        
        # Add baseline information if available
        if 'overall_baseline' in result:
            response_data['overall_baseline'] = round(result['overall_baseline'], 1)
        
        # Determine colors and icons based on feedback type
        feedback_type = result.get('feedback_type', 'normal')
        anomaly_level = result.get('anomaly_level', 'normal')
        
        if feedback_type == 'recommendation':
            if anomaly_level == 'low':
                response_data['anomaly_color'] = 'blue'
                response_data['anomaly_icon'] = '🔵'
                response_data['anomaly_text'] = 'Very Low Usage'
            elif anomaly_level == 'pattern':
                response_data['anomaly_color'] = 'orange'
                response_data['anomaly_icon'] = '🟠'
                response_data['anomaly_text'] = 'Moderate Waste'
            elif anomaly_level == 'high':
                response_data['anomaly_color'] = 'red'
                response_data['anomaly_icon'] = '🔴'
                response_data['anomaly_text'] = 'Critical Waste'
            else:
                response_data['anomaly_color'] = 'black'
                response_data['anomaly_icon'] = '⚫'
                response_data['anomaly_text'] = 'Unknown'
        else:
            # Positive feedback
            error_percentage = result.get('prediction_error_percentage', 0)
            if error_percentage < -20:
                response_data['anomaly_color'] = 'darkgreen'
                response_data['anomaly_icon'] = '🏆'
                response_data['anomaly_text'] = 'Excellent Savings'
            elif error_percentage < -10:
                response_data['anomaly_color'] = 'green'
                response_data['anomaly_icon'] = '👍'
                response_data['anomaly_text'] = 'Good Savings'
            else:
                response_data['anomaly_color'] = 'green'
                response_data['anomaly_icon'] = '✅'
                response_data['anomaly_text'] = 'Normal Operation'
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/building_info')
def building_info():
    """Get information about a specific building"""
    global predictor
    
    try:
        building_name = request.args.get('building')
        
        if not building_name or predictor is None:
            return jsonify({'error': 'Invalid request'})
        
        if building_name not in predictor.baselines:
            return jsonify({'error': 'Building not found'})
        
        baseline_info = predictor.baselines[building_name]
        
        
        response_data = {
            'building': building_name,
            
            'overall_baseline': round(baseline_info['overall']['mean'], 1),
            'weekday_baseline': round(baseline_info['by_day_type']['weekdays']['mean'], 1),
            'weekend_baseline': round(baseline_info['by_day_type']['weekends']['mean'], 1),
            'data_points': baseline_info['overall']['count']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/status')
def status():
    """Check system status"""
    global predictor
    
    if predictor is None:
        return jsonify({
            'initialized': False,
            'models_trained': 0,
            'buildings_available': [],
            'recommendations_loaded': False,
           
        })
    
    status_info = {
        'initialized': predictor is not None,
        'models_trained': len(predictor.models) if predictor else 0,
        'buildings_available': list(predictor.models.keys()) if predictor else [],
        'recommendations_loaded': predictor.recommendations_df is not None if predictor else False,
        'recommendations_count': len(predictor.recommendations_df) if predictor and predictor.recommendations_df is not None else 0,
       
    }
    
    return jsonify(status_info)

@app.route('/debug_recommendations')
def debug_recommendations():
    """Debug endpoint to check recommendations matching"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'})
    
    debug_info = {
        'recommendations_loaded': predictor.recommendations_df is not None,
        'recommendations_columns': list(predictor.recommendations_df.columns) if predictor.recommendations_df is not None else [],
        'recommendations_sample': []
    }
    
    if predictor.recommendations_df is not None and len(predictor.recommendations_df) > 0:
        building_col = predictor.recommendations_df.columns[0]
        for i in range(min(5, len(predictor.recommendations_df))):
            debug_info['recommendations_sample'].append({
                'index': i,
                'building_name': str(predictor.recommendations_df.iloc[i][building_col]),
                'recommendations': {}
            })
            # Get all non-building columns
            for col in predictor.recommendations_df.columns:
                if col != building_col:
                    value = predictor.recommendations_df.iloc[i][col]
                    if pd.notna(value):
                        debug_info['recommendations_sample'][-1]['recommendations'][col] = str(value)
    
    return jsonify(debug_info)

if __name__ == '__main__':
    # Initialize predictor on startup
    success, message = initialize_predictor()
    if success:
        print("Predictor initialized successfully")
    else:
        print(f" Predictor initialization failed: {message}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)