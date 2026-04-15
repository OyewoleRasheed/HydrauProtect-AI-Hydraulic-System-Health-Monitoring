import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

PUMP_LABELS = {
    0: 'No leakage',
    1: 'Weak leakage', 
    2: 'Severe leakage'
}

ACCUMULATOR_LABELS = {
    90: 'Close to failure',
    100: 'Severely reduced pressure',
    115: 'Slightly reduced pressure',
    130: 'Optimal pressure'
}


def load_models():
    pump_model = joblib.load(os.path.join(MODELS_DIR, 'pump_model.pkl'))
    accum_model = joblib.load(os.path.join(MODELS_DIR, 'accumulator_model.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    return pump_model, accum_model, feature_names

def predict(features_df: pd.DataFrame) -> list:
    """
    Runs both models on feature dataframe.
    Returns list of predictions per cycle.
    """
    pump_model, accum_model, feature_names = load_models()
    
    X = features_df[feature_names]
    
    pump_preds = pump_model.predict(X)
    pump_probs = pump_model.predict_proba(X)
    
    accum_preds = accum_model.predict(X)
    accum_probs = accum_model.predict_proba(X)
    
    accum_classes = accum_model.classes_
    
    results = []
    for i in range(len(X)):
        pump_pred = int(pump_preds[i])
        accum_pred = int(accum_preds[i])
        
        pump_prob_dict = {
            PUMP_LABELS[j]: round(float(pump_probs[i][j]), 3)
            for j in range(len(pump_probs[i]))
        }
        
        accum_prob_dict = {
            ACCUMULATOR_LABELS[int(accum_classes[j])]: round(float(accum_probs[i][j]), 3)
            for j in range(len(accum_probs[i]))
        }
        
        results.append({
            'cycle': i + 1,
            'pump_prediction': PUMP_LABELS[pump_pred],
            'pump_probabilities': pump_prob_dict,
            'accumulator_prediction': ACCUMULATOR_LABELS[accum_pred],
            'accumulator_probabilities': accum_prob_dict,
            'requires_attention': pump_pred > 0 or accum_pred != 130
        })
    
    return results

if __name__ == "__main__":
    from feature_engineering import generate_csv_template
    import numpy as np
    
    template = generate_csv_template()
    test_row = {col: np.random.uniform(0, 1) for col in template.columns}
    test_df = pd.DataFrame([test_row])
    
    results = predict(test_df)
    print("Test prediction:")
    print(results[0])