"""London borough environmental risk context shared by the web pages."""
import os

import pandas as pd

# Environmental risk multiplier and short description for each borough
BOROUGH_RISK = {
    'Barking and Dagenham': (1.20, 'High air pollution (PM2.5: 15 μg/m³), limited green space, high CVD mortality rate.'),
    'Tower Hamlets': (1.18, 'High NO₂ levels (45+ μg/m³), urban density, elevated cardiovascular mortality.'),
    'Hackney': (1.17, 'Above-average pollution, traffic density, higher CVD hospitalisation rates.'),
    'Newham': (1.16, 'Industrial pollution exposure, limited green space access.'),
    'City of London': (1.15, 'Very high NO₂ (87 μg/m³), traffic pollution, urban heat island effect.'),
    'Westminster': (1.14, 'Highest NO₂ in London (88 μg/m³), heavy traffic exposure.'),
    'Camden': (1.13, 'High pollution levels (82.3 μg/m³ NO₂), urban environment.'),
    'Islington': (1.08, 'Moderate pollution, limited green space.'),
    'Southwark': (1.07, 'Mixed pollution exposure, some green areas.'),
    'Lambeth': (1.06, 'Urban environment with moderate air quality.'),
    'Brent': (1.06, 'Traffic pollution from major roads.'),
    'Greenwich': (1.05, 'Some green space, moderate pollution levels.'),
    'Lewisham': (1.05, 'Outer London location, mixed environmental factors.'),
    'Hammersmith and Fulham': (1.05, 'Urban location, moderate pollution.'),
    'Ealing': (1.04, 'Better air quality, some green space.'),
    'Croydon': (1.04, 'Urban centre, some pollution exposure.'),
    'Haringey': (1.04, 'Mixed urban environment.'),
    'Kensington and Chelsea': (1.03, 'Central location but better air quality.'),
    'Merton': (1.03, 'Suburban environment, moderate pollution.'),
    'Waltham Forest': (1.03, 'Some green space, moderate pollution levels.'),
    'Hounslow': (1.03, 'Airport impact, moderate pollution levels.'),
    'Wandsworth': (1.02, 'Good green space access, moderate air quality.'),
    'Bexley': (1.02, 'Outer London, generally better air quality.'),
    'Redbridge': (1.02, 'Outer London location, moderate environmental risk.'),
    'Hillingdon': (1.02, 'Airport proximity but generally good air quality.'),
    'Bromley': (1.01, 'Lower PM2.5 (12.4 μg/m³), suburban environment.'),
    'Enfield': (1.01, 'Outer London, better environmental conditions.'),
    'Harrow': (1.01, 'Suburban location, moderate air quality.'),
    'Havering': (1.00, 'Low PM2.5 (12.1 μg/m³), rural characteristics.'),
    'Barnet': (0.98, 'Lower pollution levels, good green space access.'),
    'Sutton': (0.95, 'Suburban environment, good environmental conditions.'),
    'Kingston upon Thames': (0.90, 'Good air quality, riverside location, ample green space.'),
    'Richmond upon Thames': (0.85, 'Excellent air quality, extensive green space (Richmond Park), lowest CVD mortality.'),
}

TIERS = [
    ('high', 'Higher exposure'),
    ('medium', 'Medium exposure'),
    ('low', 'Lower exposure'),
]

ENV_CSV = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'environmental_data',
                       'expanded_environmental_data.csv')


def tier_for(multiplier):
    if multiplier >= 1.10:
        return 'high'
    if multiplier >= 1.00:
        return 'medium'
    return 'low'


def load_environment():
    return pd.read_csv(ENV_CSV)


def borough_table(env_data):
    """One row per borough: environmental readings plus risk context, highest risk first."""
    rows = []
    for rec in env_data.to_dict('records'):
        multiplier, description = BOROUGH_RISK.get(rec['Borough'], (1.0, ''))
        rows.append({
            'name': rec['Borough'],
            'pm25': round(float(rec['Avg_PM25']), 1),
            'no2': round(float(rec['Avg_NO2']), 1),
            'noise': round(float(rec['NoiseLevel_dB']), 1),
            'green': round(float(rec['GreenSpacePercent']), 1),
            'walk': round(float(rec['WalkabilityScore']), 1),
            'heat': round(float(rec['UrbanHeatIncrease']), 1),
            'multiplier': multiplier,
            'tier': tier_for(multiplier),
            'description': description,
        })
    rows.sort(key=lambda r: (-r['multiplier'], r['name']))
    return rows


def grouped_boroughs():
    """Borough names grouped by exposure tier, for the assessment form."""
    groups = {key: [] for key, _ in TIERS}
    for name, (multiplier, _) in BOROUGH_RISK.items():
        groups[tier_for(multiplier)].append(name)
    return [(label, sorted(groups[key])) for key, label in TIERS]
