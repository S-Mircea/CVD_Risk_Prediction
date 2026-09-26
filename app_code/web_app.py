import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from ml_model import CVDRiskModel
from llm_advisor import CVDLlamaAdvisor
from borough_info import BOROUGH_RISK, borough_table, grouped_boroughs, load_environment
import traceback

app = Flask(__name__)

# Initialize model
model = CVDRiskModel()
if not model.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cvd_risk_model.pkl')):
    print("Warning: Model not found. Please train the model first.")

# Initialize LLM advisor if available
try:
    llm_advisor = CVDLlamaAdvisor()
    LLM_AVAILABLE = True
    ollama_status = llm_advisor.check_ollama_availability()
    if ollama_status:
        print("✓ Ollama LLM advisor connected successfully")
    else:
        print("⚠ Ollama not available - using fallback advice system")
except Exception as e:
    print(f"⚠ LLM advisor initialization failed: {e}")
    LLM_AVAILABLE = False
    ollama_status = False

ENV_DATA = load_environment()

FEATURE_LABELS = {
    'Age': ('Age', False), 'SystolicBP': ('Systolic BP', False), 'BMI': ('BMI', False),
    'HighBloodPressure_encoded': ('High BP diagnosis', False), 'TotalCholesterol': ('Cholesterol', False),
    'PhysicalActivityLevel_encoded': ('Physical activity', False), 'SleepHours': ('Sleep', False),
    'DiastolicBP': ('Diastolic BP', False), 'Avg_NO2': ('NO₂', True), 'GreenSpacePercent': ('Green space', True),
    'Avg_PM25': ('PM2.5', True), 'WalkabilityScore': ('Walkability', True), 'Borough_encoded': ('Borough', True),
    'Diabetes_encoded': ('Diabetes', False), 'NoiseLevel_dB': ('Noise', True),
    'UrbanHeatIncrease': ('Urban heat', True), 'Smoker_encoded': ('Smoking', False),
    'FamilyHistoryCVD_encoded': ('Family history', False), 'AlcoholConsumption_encoded': ('Alcohol', False),
    'StressLevel_encoded': ('Stress', False), 'Gender_encoded': ('Sex', False),
}


def feature_importances():
    """The trained model's own feature importances, largest first."""
    if model.model is None or not hasattr(model.model, 'feature_importances_'):
        return []
    pairs = zip(model.data_processor.feature_columns, model.model.feature_importances_)
    rows = []
    for name, weight in sorted(pairs, key=lambda p: -p[1]):
        label, environmental = FEATURE_LABELS.get(name, (name, False))
        rows.append({'label': label, 'weight': round(float(weight) * 100, 1), 'env': environmental})
    return rows


@app.route('/')
def index():
    selected = request.args.get('borough', '')
    if selected not in BOROUGH_RISK:
        selected = ''
    return render_template(
        'index.html',
        active='assess',
        borough_groups=grouped_boroughs(),
        selected_borough=selected,
        borough_risk={name: {'multiplier': m, 'description': d} for name, (m, d) in BOROUGH_RISK.items()},
        london_avg={'pm25': round(float(ENV_DATA['Avg_PM25'].mean()), 1),
                    'no2': round(float(ENV_DATA['Avg_NO2'].mean()), 1)},
    )


@app.route('/how-it-works')
def how_it_works():
    return render_template('how.html', active='how', importances=feature_importances())


@app.route('/boroughs')
def boroughs():
    rows = borough_table(ENV_DATA)
    return render_template('boroughs.html', active='boroughs', boroughs=rows)


@app.route('/about')
def about():
    return render_template('about.html', active='about')

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    try:
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
        
        env_data = ENV_DATA
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
        
        result = model.predict_risk(user_data)
        result['environmental_data'] = {
            'pm25': user_data['Avg_PM25'],
            'no2': user_data['Avg_NO2'],
            'borough': user_data['Borough'],
            'green_space': user_data['GreenSpacePercent']
        }
        result['recommendations'] = get_recommendations(result['risk_level'])
        
        # Generate LLM-powered environmental advice
        if LLM_AVAILABLE and ollama_status:
            try:
                advice = llm_advisor.get_environmental_advice(
                    result['risk_level'], result['environmental_data'], user_data
                )
                print("Llama advice:", advice)
                result['llm_advice'] = advice
                result['llm_available'] = True
            except Exception as e:
                print("Llama error:", e)
                traceback.print_exc()
                advice = "Sorry, no advice available at this time."
                result['llm_advice'] = advice
                result['llm_available'] = False
        else:
            result['llm_advice'] = None
            result['llm_available'] = False
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        print("Error in assess_risk:", e)
        traceback.print_exc()
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

@app.route('/llm-status', methods=['GET'])
def llm_status():
    if not LLM_AVAILABLE:
        return jsonify({
            'success': True,
            'status': {
                'model_name': 'Not Available',
                'available': False,
                'ollama_running': False
            }
        })
    try:
        model_info = llm_advisor.get_model_info()
        return jsonify({
            'success': True,
            'status': model_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting CVD Risk Assessment application...")
    print("Server will be available at:")
    print("- http://127.0.0.1:5002")
    print("- http://localhost:5002")
    print("Press Ctrl+C to stop the server")
    try:
        app.run(debug=True, host='127.0.0.1', port=5002)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Trying alternative configuration...")
        app.run(debug=False, host='localhost', port=5002)