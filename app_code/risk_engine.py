"""10-year cardiovascular risk from the Framingham general CVD equation.

Base risk: D'Agostino RB Sr et al. "General cardiovascular risk profile for use in
primary care: the Framingham Heart Study." Circulation 2008;117:743-753.
Validated for ages 30-74 without existing cardiovascular disease.

The London borough multiplier is applied on top as an illustrative environmental
adjustment. It scales the hazard, so 1 - (1 - risk) ** multiplier keeps the
result between 0 and 1. It is not part of the validated equation.
"""
import math

# Sex-specific coefficients, baseline survival and mean linear predictor (Table 2 of the paper)
COEFFICIENTS = {
    'Female': {'age': 2.32888, 'tc': 1.20904, 'hdl': -0.70833, 'sbp_untreated': 2.76157,
               'sbp_treated': 2.82263, 'smoker': 0.52873, 'diabetes': 0.69154,
               's0': 0.95012, 'mean': 26.1931},
    'Male': {'age': 3.06117, 'tc': 1.12370, 'hdl': -0.93263, 'sbp_untreated': 1.93303,
             'sbp_treated': 1.99881, 'smoker': 0.65451, 'diabetes': 0.57367,
             's0': 0.88936, 'mean': 23.9802},
}

MIN_AGE, MAX_AGE = 30, 74
MMOL_TO_MG_DL = 38.67

# Healthy reference values used to show how much each factor adds
REFERENCE = {'sbp': 120, 'tc': 170, 'hdl': 60}

# Clinical bands: NICE (CG181/NG238) offers statins from 10% 10-year risk
LEVELS = [(0.05, 'Low Risk'), (0.10, 'Moderate Risk'), (0.20, 'High Risk'), (float('inf'), 'Very High Risk')]


def cholesterol_to_mg_dl(value):
    """Accept mg/dL or mmol/L: values under 20 can only be mmol/L."""
    value = float(value)
    return value * MMOL_TO_MG_DL if value < 20 else value


def framingham_risk(age, sex, total_chol, hdl, sbp, bp_treated, smoker, diabetic):
    """10-year risk of a first cardiovascular event, as a fraction."""
    c = COEFFICIENTS['Female' if sex == 'Female' else 'Male']
    x = (c['age'] * math.log(age)
         + c['tc'] * math.log(total_chol)
         + c['hdl'] * math.log(hdl)
         + (c['sbp_treated'] if bp_treated else c['sbp_untreated']) * math.log(sbp)
         + c['smoker'] * smoker
         + c['diabetes'] * diabetic)
    return 1 - c['s0'] ** math.exp(x - c['mean'])


def adjust_for_environment(risk, multiplier):
    return 1 - (1 - risk) ** multiplier


def risk_level(risk):
    return next(label for limit, label in LEVELS if risk < limit)


def assess(age, sex, total_chol, hdl, sbp, bp_treated, smoker, diabetic, env_multiplier=1.0):
    """Full assessment: base and adjusted risk, level, and what each factor adds."""
    age_used = min(max(age, MIN_AGE), MAX_AGE)
    inputs = dict(age=age_used, sex=sex, total_chol=total_chol, hdl=hdl, sbp=sbp,
                  bp_treated=bp_treated, smoker=int(smoker), diabetic=int(diabetic))

    def total(**changes):
        params = {**inputs, **{k: v for k, v in changes.items() if k != 'env'}}
        return adjust_for_environment(framingham_risk(**params), changes.get('env', env_multiplier))

    base = framingham_risk(**inputs)
    risk = adjust_for_environment(base, env_multiplier)

    # Percentage points each factor adds compared with a healthy reference value
    drivers = [
        ('Age', risk - total(age=MIN_AGE)),
        ('Blood pressure', risk - total(sbp=min(sbp, REFERENCE['sbp']), bp_treated=False)),
        ('Cholesterol', risk - total(total_chol=min(total_chol, REFERENCE['tc']))),
        ('HDL cholesterol', risk - total(hdl=max(hdl, REFERENCE['hdl']))),
        ('Smoking', risk - total(smoker=0)),
        ('Diabetes', risk - total(diabetic=0)),
        ('Environment', risk - total(env=1.0)),
    ]

    note = None
    if age < MIN_AGE:
        note = f'The equation is validated from age {MIN_AGE}, so this shows your risk as if you were {MIN_AGE}. Your real 10-year risk is likely lower.'
    elif age > MAX_AGE:
        note = f'The equation is validated up to age {MAX_AGE}, so this shows your risk as if you were {MAX_AGE}. Your real 10-year risk is likely higher.'

    return {
        'risk_probability': risk,
        'base_risk': base,
        'risk_level': risk_level(risk),
        'drivers': [{'name': name, 'points': round(points * 100, 1)} for name, points in drivers],
        'age_note': note,
    }
