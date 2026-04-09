import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

PUMP_SEVERITY = {
    'No leakage': 'low',
    'Weak leakage': 'medium',
    'Severe leakage': 'high'
}

ACCUMULATOR_SEVERITY = {
    'Optimal pressure': 'low',
    'Slightly reduced pressure': 'medium',
    'Severely reduced pressure': 'high',
    'Close to failure': 'critical'
}

def build_prompt(prediction_result: dict, shap_explanation: dict) -> str:
    pump_pred = prediction_result['pump_prediction']
    pump_probs = prediction_result['pump_probabilities']
    accum_pred = prediction_result['accumulator_prediction']
    accum_probs = prediction_result['accumulator_probabilities']
    contributors = shap_explanation['top_contributors']

    contributor_text = "\n".join([
        f"- {c['description']}: {c['direction']} "
        f"(magnitude: {c['magnitude']}, actual value: {c['actual_value']})"
        for c in contributors
    ])

    prompt = f"""You are an AI assistant for industrial hydraulic system maintenance in oil and gas facilities.

A hydraulic system health model has assessed one operating cycle and returned these findings:

PUMP LEAKAGE STATUS: {pump_pred}
This is the CONFIRMED diagnosis. Use exactly this severity level in your report.
Highest confidence: {max(pump_probs, key=pump_probs.get)} at {max(pump_probs.values())*100:.0f}%

ACCUMULATOR CONDITION: {accum_pred}
Highest confidence: {max(accum_probs, key=accum_probs.get)} at {max(accum_probs.values())*100:.0f}%


TOP SENSOR CONTRIBUTORS TO PUMP ASSESSMENT:
{contributor_text}

Write a maintenance finding report in plain English for a shift engineer.
Your response must follow this exact structure:

FINDING:
One sentence stating what was detected and severity.

SENSOR EVIDENCE:
Two to three sentences explaining which sensors flagged the issue and what their readings indicate physically. Use plain engineering language, not data science language. Do not mention SHAP values or model probabilities directly.

CONSEQUENCE OF DEFERRAL:
One sentence stating what happens if this is not addressed.

RECOMMENDED ACTION:
One clear action the engineer should take, with urgency level (immediate, within 24 hours, next scheduled maintenance window).

Keep the entire response under 200 words. Write for a maintenance engineer, not a data scientist."""

    return prompt


def generate_narrative(prediction_result: dict,
                       shap_explanation: dict) -> str:
    """
    Calls Groq API and returns plain English maintenance finding.
    """
    if not GROQ_API_KEY:
        return generate_fallback_narrative(prediction_result)

    prompt = build_prompt(prediction_result, shap_explanation)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama-3.1-8b-instant",
        "max_tokens": 400,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert hydraulic systems maintenance engineer in the oil and gas industry. You write clear, concise maintenance reports."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            GROQ_URL,
            headers=headers,
            json=body,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']

    except Exception as e:
        print(f"Groq API error: {e}")
        return generate_fallback_narrative(prediction_result)


def generate_fallback_narrative(prediction_result: dict) -> str:
    pump = prediction_result['pump_prediction']
    accum = prediction_result['accumulator_prediction']

    pump_severity = PUMP_SEVERITY.get(pump, 'unknown')
    accum_severity = ACCUMULATOR_SEVERITY.get(accum, 'unknown')

    if pump_severity == 'high' or accum_severity == 'critical':
        urgency = "Immediate attention required."
    elif pump_severity == 'medium' or accum_severity == 'high':
        urgency = "Address within 24 hours."
    else:
        urgency = "Monitor at next scheduled maintenance window."

    return f"""FINDING:
Hydraulic system assessment detected {pump.lower()} in pump and {accum.lower()} in accumulator.

SENSOR EVIDENCE:
Motor power and flow sensors indicate deviation from healthy baseline.
Pressure sensor readings show abnormal operating pattern consistent with detected fault condition.

CONSEQUENCE OF DEFERRAL:
Unaddressed leakage progresses from weak to severe, risking unplanned shutdown.

RECOMMENDED ACTION:
{urgency} Inspect pump seals and accumulator charge pressure.
Schedule maintenance work order referencing ISO 4413 section 5.4."""


if __name__ == "__main__":
    test_prediction = {
        'cycle': 1,
        'pump_prediction': 'Weak leakage',
        'pump_probabilities': {
            'No leakage': 0.18,
            'Weak leakage': 0.71,
            'Severe leakage': 0.11
        },
        'accumulator_prediction': 'Slightly reduced pressure',
        'accumulator_probabilities': {
            'Optimal pressure': 0.12,
            'Slightly reduced pressure': 0.68,
            'Severely reduced pressure': 0.14,
            'Close to failure': 0.06
        },
        'requires_attention': True
    }

    test_shap = {
        'top_contributors': [
            {
                'feature': 'EPS1_mean',
                'description': 'motor power (average)',
                'shap_value': 0.041,
                'actual_value': 2341.5,
                'direction': 'increases risk',
                'magnitude': 'high'
            },
            {
                'feature': 'FS1_mean',
                'description': 'volume flow 1 (average)',
                'shap_value': 0.029,
                'actual_value': 8.3,
                'direction': 'increases risk',
                'magnitude': 'moderate'
            }
        ],
        'model_type': 'pump'
    }

    print("Testing Groq narrative generation...")
    narrative = generate_narrative(test_prediction, test_shap)
    print("\n" + narrative)