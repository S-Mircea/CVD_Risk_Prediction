# Heartscape: CVD Risk Prediction App

A cardiovascular disease risk web app that combines the validated Framingham risk equation with London-specific environmental factors.

## Project Overview

This application addresses the limitations of traditional CVD risk calculators (Framingham, QRISK) by leveraging machine learning to provide more personalized and accurate risk assessments. Unlike conventional methods, our app integrates environmental data specific to London boroughs, making it particularly relevant for UK populations.

### Key Features

- **Validated base score**: the Framingham general CVD equation (D'Agostino et al., Circulation 2008), checked against its published worked examples
- **Environmental adjustment**: an illustrative borough multiplier (×0.85–×1.20) applied as a hazard scaling
- **Explainable results**: percentage points each factor adds, clinical risk bands (<5%, 5–10%, 10–20%, 20%+)
- **Personal guidance** from all answers, including lifestyle factors the equation leaves out
- **Borough data explorer** for all 33 London boroughs

## How the Score Works

`app_code/risk_engine.py` computes the 10-year risk from age, sex, total and HDL cholesterol (mg/dL or mmol/L),
systolic blood pressure (treated or untreated), smoking and diabetes, then applies `1 - (1 - risk) ** multiplier`
for the chosen borough. It is validated for ages 30–74 without existing cardiovascular disease.

The earlier Random Forest model (`ml_model.py`, `cvd_risk_model.pkl`) is kept for reference but no longer used:
its synthetic training data labelled 89% of people, and everyone over 40, as at risk, so it scored almost all
adults as "Very High Risk".

This is an educational tool, not a diagnostic device.

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/S-Mircea/CVD_Risk_Prediction.git
   cd CVD_Risk_Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r app_code/requirements.txt
   ```

3. **Run the application**
   ```bash
   cd app_code
   python3 web_app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://127.0.0.1:5002`
   - The application will be ready to use!

## Usage Guide

### Patient Data Input

The application requires the following information:

**Personal Information:**
- Age (18-100 years)
- Gender (Male/Female)

**Medical History:**
- Family history of CVD
- Diabetes status
- High blood pressure
- Smoking status

**Lifestyle Factors:**
- Physical activity level
- Alcohol consumption
- Stress level
- Sleep hours per night

**Clinical Measurements:**
- BMI (Body Mass Index)
- Total cholesterol (mg/dL)
- Systolic blood pressure (mmHg)
- Diastolic blood pressure (mmHg)

**Location:**
- London borough (for environmental risk assessment)

### Risk Assessment

After submitting the form, the application provides:

1. **Visual Risk Display**: Animated heart showing risk percentage
2. **Risk Classification**: Low/Moderate/High risk categories
3. **Environmental Factors**: Borough-specific pollution and environmental data
4. **Feature Importance**: Bar chart showing key contributing factors
5. **Personalized Recommendations**: Tailored advice based on risk level

## Project Structure

```
CVD_Risk_Prediction/
├── app_code/
│   ├── templates/
│   │   ├── base.html           # Shared layout: header, navigation, footer
│   │   ├── index.html          # Risk assessment form and results
│   │   ├── how.html            # How it works, with live model feature importances
│   │   ├── boroughs.html       # Sortable environmental data for all 33 boroughs
│   │   └── about.html          # Background, limitations and privacy
│   ├── static/                 # app.css, assess.js, boroughs.js
│   ├── borough_info.py         # Borough risk multipliers and table helpers
│   ├── web_app.py              # Flask application
│   ├── ml_model.py             # Machine learning model class
│   ├── data_processor.py       # Data preprocessing utilities
│   ├── train_model.py          # Model training script
│   ├── cvd_risk_model.pkl      # Trained model file
│   └── requirements.txt        # Python dependencies
├── data/
├── environmental_data/
│   ├── expanded_environmental_data.csv  # London borough environmental data
│   └── london_environmental_data.csv
├── user_data/
│   ├── expanded_health_data.csv         # Training health data
│   └── personal_health_data.csv
└── README.md
```

## Environmental Data Integration

The application incorporates London-specific environmental factors:

- **Air Quality**: PM2.5 and NO2 pollution levels
- **Urban Environment**: Noise levels, green space percentage
- **Lifestyle Factors**: Walkability scores, urban heat island effects

Data covers all 33 London boroughs with risk multipliers based on environmental conditions.

## Machine Learning Details

### Algorithm: Random Forest Classifier

**Model Configuration:**
- 200 decision trees
- Maximum depth: 15
- Balanced class weights
- 5-fold cross-validation

**Feature Engineering:**
- Label encoding for categorical variables
- Standard scaling for numerical features
- Environmental data merging by borough

### Training Data

- **Size**: 1,000 synthetic patient records
- **Target**: Binary CVD risk (0/1)
- **Features**: 21 engineered features
- **Split**: 80% training, 20% testing
- **Validation**: Stratified cross-validation

## Future Enhancements

### Planned Features

- **Real-time Environmental Data**: Integration with live London air quality APIs
- **Enhanced Genetic Profiling**: Detailed family history and genetic markers
- **Adaptive Learning**: Model updates based on new research and data
- **User Accounts**: Risk tracking over time
- **Clinical Validation**: Partnership with healthcare providers

### Technical Improvements

- **API Integration**: LAQN (London Air Quality Network) real-time data
- **Database Enhancement**: PostgreSQL for production deployment
- **Cloud Deployment**: AWS/Azure hosting with auto-scaling
- **Mobile Application**: React Native mobile version

## Research Background

This project addresses key limitations identified in cardiovascular risk assessment:

- Traditional methods (Framingham, QRISK) ignore diverse populations and modern lifestyles
- Environmental factors significantly impact CVD risk but are often overlooked
- Machine learning can capture complex, non-linear relationships in health data
- London's air quality issues make environmental integration particularly relevant

## Medical Disclaimer

**Important**: This application is designed for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Mircea Serban**
- GitHub: [@S-Mircea](https://github.com/S-Mircea)
- Project: Final Year Project - CVD Risk Assessment

## Acknowledgments

- London Air Quality Network for environmental data
- scikit-learn community for machine learning tools
- Flask framework for web development
- London borough environmental research initiatives

---

*Built with love for better cardiovascular health awareness*
