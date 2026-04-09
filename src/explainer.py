import shap
import numpy as np
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

SENSOR_DESCRIPTIONS = {
    'PS1': 'system pressure 1',
    'PS2': 'system pressure 2',
    'PS3': 'system pressure 3',
    'PS4': 'system pressure 4',
    'PS5': 'system pressure 5',
    'PS6': 'return line pressure',
    'EPS1': 'motor power',
    'FS1': 'volume flow 1',
    'FS2': 'volume flow 2',
    'TS1': 'temperature 1',
    'TS2': 'temperature 2',
    'TS3': 'temperature 3',
    'TS4': 'temperature 4',
    'VS1': 'vibration',
    'SE': 'system efficiency',
    'CE': 'cooling efficiency'
}

def get_sensor_description(feature_name: str) -> str:
    parts = feature_name.rsplit('_', 1)
    sensor = parts[0]
    stat = parts[1] if len(parts) > 1 else ''
    
    stat_map = {
        'mean': 'average',
        'std': 'variability',
        'min': 'minimum',
        'max': 'maximum',
        'range': 'operating range'
    }
    
    sensor_desc = SENSOR_DESCRIPTIONS.get(sensor, sensor)
    stat_desc = stat_map.get(stat, stat)
    
    return f"{sensor_desc} ({stat_desc})"

def explain_prediction(features_df: pd.DataFrame, 
                       predicted_class_idx: int,
                       model_type: str = 'pump') -> dict:
    """
    Returns top 5 SHAP contributors for a single cycle prediction.
    
    model_type: 'pump' or 'accumulator'
    predicted_class_idx: index of predicted class
    """
    model_path = os.path.join(
        MODELS_DIR, 
        'pump_model.pkl' if model_type == 'pump' else 'accumulator_model.pkl'
    )
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    model = joblib.load(model_path)
    
    X = features_df[feature_names]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle both Random Forest and XGBoost SHAP output formats
    print("SHAP values type:", type(shap_values))
    print("SHAP values length:", len(shap_values))

    if isinstance(shap_values, list):
        # Random Forest returns list of arrays, one per class
        shap_for_class = shap_values[predicted_class_idx][0]
    else:
        # XGBoost/newer SHAP returns 3D array (samples, features, classes)
        shap_for_class = shap_values[0, :, predicted_class_idx]
        
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X)
    
    # shap_for_class = shap_values[predicted_class_idx][0]
    
    top_indices = np.argsort(np.abs(shap_for_class))[-5:][::-1]
    
    contributors = []
    for idx in top_indices:
        feature = feature_names[idx]
        value = float(shap_for_class[idx])
        actual_value = float(X.iloc[0][feature])
        
        contributors.append({
            'feature': feature,
            'description': get_sensor_description(feature),
            'shap_value': round(value, 4),
            'actual_value': round(actual_value, 4),
            'direction': 'increases risk' if value > 0 else 'decreases risk',
            'magnitude': 'high' if abs(value) > 0.05 else 'moderate' 
                        if abs(value) > 0.02 else 'low'
        })
    
    return {
        'top_contributors': contributors,
        'model_type': model_type
    }

if __name__ == "__main__":
    print("Explainer module loaded successfully")
    print("Sensor descriptions available for:", list(SENSOR_DESCRIPTIONS.keys()))