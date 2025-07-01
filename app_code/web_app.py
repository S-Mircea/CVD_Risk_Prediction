from flask import Flask, render_template, request, jsonify
import pandas as pd
from ml_model import CVDRiskModel
import os

app = Flask(__name__)

# Initialize model
model = CVDRiskModel()
if not model.load_model():
    print("Warning: Model not found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    try:
        # Get form data
        user_data = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Smoker': request.form['smoker'],
            'FamilyHistoryCVD': request.form['family_history'],
            'Diabetes': request.form['diabetes'],
            'HighBloodPressure': request.form['high_bp'],
            'PhysicalActivityLevel': request.form['activity'],
            'BMI': float(request.form['bmi']),
            'TotalCholesterol': float(request.form['cholesterol']),
            'SystolicBP': float(request.form['systolic_bp']),
            'DiastolicBP': float(request.form['diastolic_bp']),
            'AlcoholConsumption': request.form['alcohol'],
            'StressLevel': request.form['stress'],
            'SleepHours': float(request.form['sleep_hours']),
            'Borough': request.form['borough']
        }
        
        # Load environmental data for the borough
        env_data = pd.read_csv('../environmental_data/expanded_environmental_data.csv')
        borough_env = env_data[env_data['Borough'] == user_data['Borough']]
        
        if not borough_env.empty:
            user_data['Avg_PM25'] = borough_env['Avg_PM25'].iloc[0]
            user_data['Avg_NO2'] = borough_env['Avg_NO2'].iloc[0]
            user_data['NoiseLevel_dB'] = borough_env['NoiseLevel_dB'].iloc[0]
            user_data['GreenSpacePercent'] = borough_env['GreenSpacePercent'].iloc[0]
            user_data['WalkabilityScore'] = borough_env['WalkabilityScore'].iloc[0]
            user_data['UrbanHeatIncrease'] = borough_env['UrbanHeatIncrease'].iloc[0]
        else:
            user_data['Avg_PM25'] = env_data['Avg_PM25'].mean()
            user_data['Avg_NO2'] = env_data['Avg_NO2'].mean()
            user_data['NoiseLevel_dB'] = env_data['NoiseLevel_dB'].mean()
            user_data['GreenSpacePercent'] = env_data['GreenSpacePercent'].mean()
            user_data['WalkabilityScore'] = env_data['WalkabilityScore'].mean()
            user_data['UrbanHeatIncrease'] = env_data['UrbanHeatIncrease'].mean()
        
        # Make prediction
        result = model.predict_risk(user_data)
        
        # Add environmental data to result
        result['environmental_data'] = {
            'pm25': user_data['Avg_PM25'],
            'no2': user_data['Avg_NO2'],
            'borough': user_data['Borough']
        }
        
        # Add recommendations
        result['recommendations'] = get_recommendations(result['risk_level'])
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def get_recommendations(risk_level):
    recommendations = {
        'Low Risk': [
            "Continue your healthy lifestyle",
            "Schedule regular check-ups",
            "Monitor environmental exposure",
            "Maintain current activity levels"
        ],
        'Moderate Risk': [
            "Increase physical activity to 150+ minutes/week",
            "Consider lifestyle modifications",
            "Consult with your healthcare provider",
            "Monitor air quality in your area",
            "Consider dietary improvements"
        ],
        'High Risk': [
            "Seek immediate medical consultation",
            "Comprehensive cardiovascular assessment needed",
            "Urgent lifestyle intervention required",
            "Consider relocation if air quality is poor",
            "Regular monitoring and follow-up essential"
        ]
    }
    return recommendations.get(risk_level, ["Consult with healthcare provider"])

if __name__ == '__main__':
    print("Starting CVD Risk Assessment application...")
    print("Server will be available at:")
    print("- http://127.0.0.1:5002")
    print("- http://localhost:5002") 
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5002)