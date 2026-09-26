import os
import traceback

from flask import Flask, render_template, request, jsonify

from borough_info import BOROUGH_RISK, borough_table, grouped_boroughs, load_environment
from llm_advisor import CVDLlamaAdvisor
from risk_engine import assess, cholesterol_to_mg_dl

app = Flask(__name__)

# Optional local LLM advice through Ollama; the site works without it
try:
    llm_advisor = CVDLlamaAdvisor()
    LLM_AVAILABLE = True
    ollama_status = llm_advisor.check_ollama_availability()
    if ollama_status:
        print("✓ Ollama LLM advisor connected successfully")
    else:
        print("⚠ Ollama not available - using built-in guidance")
except Exception as e:
    print(f"⚠ LLM advisor initialization failed: {e}")
    LLM_AVAILABLE = False
    ollama_status = False

ENV_DATA = load_environment()


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
    return render_template('how.html', active='how')


@app.route('/boroughs')
def boroughs():
    rows = borough_table(ENV_DATA)
    return render_template('boroughs.html', active='boroughs', boroughs=rows)


@app.route('/about')
def about():
    return render_template('about.html', active='about')


def personal_recommendations(user, result):
    """Actions tailored to the answers given, most important first."""
    recs = []
    level = result['risk_level']
    if level in ('High Risk', 'Very High Risk'):
        recs.append("Book a cardiovascular check with your GP and take this result with you. "
                    "NICE guidance offers statins from a 10% 10-year risk.")
    if user['Smoker'] == 'Yes':
        recs.append("Stopping smoking is the single biggest change you can make; the NHS Quit Smoking service is free.")
    if user['SystolicBP'] >= 140 or user['DiastolicBP'] >= 90:
        recs.append("Your blood pressure is in the high range (140/90 or above). Have it rechecked and discuss it with your GP.")
    elif user['SystolicBP'] >= 130:
        recs.append("Your blood pressure is slightly raised. Less salt, more activity and less alcohol all help bring it down.")
    if user['TotalCholesterolMgDl'] >= 240:
        recs.append("Your total cholesterol is high. A fasting lipid test and diet review are worthwhile.")
    if user['HDLMgDl'] < 40:
        recs.append("Your HDL (good) cholesterol is low. Regular aerobic exercise is the most reliable way to raise it.")
    if user['Diabetes'] == 'Yes':
        recs.append("Keeping blood sugar in your target range lowers cardiovascular risk; keep up regular diabetes reviews.")
    if user['PhysicalActivityLevel'] == 'Low':
        recs.append("Build up to 150 minutes of moderate activity a week; brisk walking counts.")
    if user['BMI'] >= 30:
        recs.append("Losing 5–10% of your body weight measurably improves blood pressure and cholesterol.")
    if user['AlcoholConsumption'] == 'Heavy':
        recs.append("Keep alcohol under 14 units a week, spread over several days.")
    if user['SleepHours'] < 6:
        recs.append("Aim for 7–9 hours of sleep; short sleep is linked to higher blood pressure.")
    if user['FamilyHistoryCVD'] == 'Yes':
        recs.append("A family history of heart disease raises risk beyond this estimate. Mention it to your GP.")
    multiplier = BOROUGH_RISK.get(user['Borough'], (1.0, ''))[0]
    if multiplier >= 1.10:
        recs.append("Your borough has higher air pollution: favour quieter streets for walks and check the London Air forecast on high-pollution days.")
    if not recs:
        recs.append("Your answers show no major modifiable risk factors. Keep it up and re-check every five years.")
    return recs


@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    try:
        f = request.form
        user_data = {
            'Age': int(f['age']),
            'Gender': f['gender'],
            'Smoker': f['smoker'],
            'FamilyHistoryCVD': f['family_history'],
            'Diabetes': f['diabetes'],
            'HighBloodPressure': f['high_bp'],
            'PhysicalActivityLevel': f['activity'],
            'BMI': float(f['bmi']),
            'TotalCholesterolMgDl': cholesterol_to_mg_dl(f['cholesterol']),
            'HDLMgDl': cholesterol_to_mg_dl(f['hdl']) if f.get('hdl') else 50.0,
            'SystolicBP': float(f['systolic_bp']),
            'DiastolicBP': float(f['diastolic_bp']),
            'AlcoholConsumption': f['alcohol'],
            'StressLevel': f['stress'],
            'SleepHours': float(f['sleep_hours']),
            'Borough': f['borough'],
        }

        multiplier = BOROUGH_RISK.get(user_data['Borough'], (1.0, ''))[0]
        result = assess(
            age=user_data['Age'],
            sex=user_data['Gender'],
            total_chol=user_data['TotalCholesterolMgDl'],
            hdl=user_data['HDLMgDl'],
            sbp=user_data['SystolicBP'],
            bp_treated=user_data['HighBloodPressure'] == 'Yes',
            smoker=user_data['Smoker'] == 'Yes',
            diabetic=user_data['Diabetes'] == 'Yes',
            env_multiplier=multiplier,
        )
        result['hdl_assumed'] = not f.get('hdl')

        env_row = ENV_DATA[ENV_DATA['Borough'] == user_data['Borough']]
        env = env_row.iloc[0] if not env_row.empty else ENV_DATA.mean(numeric_only=True)
        result['environmental_data'] = {
            'borough': user_data['Borough'],
            'multiplier': multiplier,
            'pm25': float(env['Avg_PM25']),
            'no2': float(env['Avg_NO2']),
            'green_space': float(env['GreenSpacePercent']),
        }
        result['recommendations'] = personal_recommendations(user_data, result)

        result['llm_advice'] = None
        result['llm_available'] = False
        if LLM_AVAILABLE and ollama_status:
            try:
                result['llm_advice'] = llm_advisor.get_environmental_advice(
                    result['risk_level'], result['environmental_data'], user_data)
                result['llm_available'] = True
            except Exception as e:
                print("Llama error:", e)

        return jsonify({'success': True, 'result': result})

    except (KeyError, ValueError) as e:
        return jsonify({'success': False, 'error': f'Please check your answers ({e}).'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/llm-status', methods=['GET'])
def llm_status():
    if not LLM_AVAILABLE:
        return jsonify({'success': True, 'status': {'model_name': 'Not Available', 'available': False,
                                                    'ollama_running': False}})
    try:
        return jsonify({'success': True, 'status': llm_advisor.get_model_info()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Heartscape on http://127.0.0.1:5002")
    app.run(debug=True, host='127.0.0.1', port=5002)
