import os
import requests
from dotenv import load_dotenv
    
import time

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
    pump_pred  = prediction_result['pump_prediction']
    pump_probs = prediction_result['pump_probabilities']
    accum_pred  = prediction_result['accumulator_prediction']
    accum_probs = prediction_result['accumulator_probabilities']
    contributors = shap_explanation['top_contributors']

    # ── Severity escalation risk ───────────────────────────────────────────
    # Pull the probability of the worst outcome for each component.
    # This is what makes recommendations differ between cycles with the same
    # fault class — a cycle where severe leakage prob is 28% is more urgent
    # than one where it's 4%, even if both are classified "Weak leakage".
    severe_leak_prob  = pump_probs.get('Severe leakage', 0.0)
    failure_prob      = accum_probs.get('Close to failure', 0.0)
    confirmed_prob    = pump_probs.get(pump_pred, 0.0)

    if severe_leak_prob >= 0.25 or failure_prob >= 0.20:
        escalation_risk = "HIGH — significant probability of rapid deterioration"
        urgency_guidance = "recommend IMMEDIATE action, do not defer"
    elif severe_leak_prob >= 0.10 or failure_prob >= 0.10:
        escalation_risk = "MODERATE — deterioration possible within days"
        urgency_guidance = "recommend action within 24 hours"
    else:
        escalation_risk = "LOW — condition appears stable for now"
        urgency_guidance = "recommend next scheduled maintenance window"

    # ── Sensor contributor text ────────────────────────────────────────────
    # Include actual sensor values AND shap influence so the LLM can reason
    # about which sensors are most deviated — producing cycle-specific findings.
    contributor_lines = []
    for c in contributors:
        line = (
            f"- {c['description']}: {c['direction']} "
            f"(SHAP influence: {c['shap_value']:+.4f}, "
            f"actual reading: {c['actual_value']:.4f}, "
            f"magnitude: {c['magnitude']})"
        )
        contributor_lines.append(line)
    contributor_text = "\n".join(contributor_lines)

    # Count how many sensors are high magnitude — used to calibrate urgency
    high_magnitude_count = sum(
        1 for c in contributors if c['magnitude'] == 'high'
    )

    prompt = f"""You are an AI assistant for industrial hydraulic system maintenance in oil and gas facilities.

A hydraulic system health model has assessed one operating cycle and returned these findings:

PUMP LEAKAGE STATUS: {pump_pred} (confirmed, {confirmed_prob*100:.0f}% model confidence)
Probability of escalating to Severe leakage: {severe_leak_prob*100:.0f}%

ACCUMULATOR CONDITION: {accum_pred}
Probability of Close to failure: {failure_prob*100:.0f}%

ESCALATION RISK: {escalation_risk}

TOP SENSOR CONTRIBUTORS (ranked by influence on this specific cycle):
{contributor_text}

SENSOR CONTEXT:
- {high_magnitude_count} out of {len(contributors)} top sensors flagged at HIGH magnitude
- Higher SHAP influence values indicate stronger deviation from healthy baseline
- Positive SHAP values push toward fault prediction; larger absolute values = stronger evidence

Write a maintenance finding report in plain English for a shift engineer.
Your response must follow this exact structure:

FINDING:
One sentence stating what was detected, severity, and confidence level.

SENSOR EVIDENCE:
Two to three sentences explaining which specific sensors are most deviated and 
what their readings indicate physically about the fault mechanism. 
Reference the actual sensor readings and what the pattern suggests.
Use plain engineering language — no mention of SHAP, model probabilities, or data science terms.

CONSEQUENCE OF DEFERRAL:
One sentence stating what happens if this is not addressed, calibrated to the 
escalation risk level above. High escalation risk = more urgent consequence.

RECOMMENDED ACTION:
One clear action with urgency level based on the escalation risk: 
{urgency_guidance}.
If {high_magnitude_count} or more sensors are at high magnitude, reflect that 
in the urgency. Be specific — name what to inspect, not just "check the system".

Keep the entire response under 220 words. 
IMPORTANT: This report must reflect the specific sensor readings and escalation 
probability for THIS cycle — do not give a generic recommendation that would 
apply to any {pump_pred} case."""

    return prompt


def generate_narrative(prediction_result: dict,
                       shap_explanation: dict) -> str:
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
        "temperature": 0.4,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert hydraulic systems maintenance engineer "
                    "in the oil and gas industry. You write clear, concise "
                    "maintenance reports. Each report must be specific to the "
                    "sensor evidence provided — never give identical "
                    "recommendations for different cycles."
                )
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
        
        # If rate limited, wait and retry once
        if response.status_code == 429:
            time.sleep(3)
            response = requests.post(
                GROQ_URL,
                headers=headers,
                json=body,
                timeout=30
            )

        response.raise_for_status()
        data = response.json()
        
        # Small delay to avoid hitting rate limit on next call
        time.sleep(2)
        
        return data['choices'][0]['message']['content']

    except Exception as e:
        print(f"Groq API error: {e}")
        return generate_fallback_narrative(prediction_result)


def generate_fallback_narrative(prediction_result: dict) -> str:
    """
    Rule-based fallback when Groq API is unavailable.
    Uses severity + accumulator state to vary urgency.
    """
    pump  = prediction_result['pump_prediction']
    accum = prediction_result['accumulator_prediction']
    pump_probs  = prediction_result.get('pump_probabilities', {})
    accum_probs = prediction_result.get('accumulator_probabilities', {})

    pump_severity  = PUMP_SEVERITY.get(pump, 'unknown')
    accum_severity = ACCUMULATOR_SEVERITY.get(accum, 'unknown')

    severe_prob  = pump_probs.get('Severe leakage', 0.0)
    failure_prob = accum_probs.get('Close to failure', 0.0)

    # Calibrate urgency to this cycle's escalation probability
    if pump_severity == 'high' or accum_severity == 'critical':
        urgency = "Immediate attention required — do not defer to next shift."
        consequence = "Continued operation risks catastrophic seal failure and unplanned shutdown."
    elif severe_prob >= 0.20 or failure_prob >= 0.15:
        urgency = "Address within 4 hours — escalation risk is elevated for this cycle."
        consequence = "Elevated probability of rapid deterioration to severe fault condition."
    elif pump_severity == 'medium' or accum_severity == 'high':
        urgency = "Address within 24 hours."
        consequence = "Unaddressed leakage progresses from weak to severe, risking unplanned shutdown."
    else:
        urgency = "Monitor at next scheduled maintenance window."
        consequence = "Condition is stable but requires trending to detect progression."

    return f"""FINDING:
Hydraulic system assessment detected {pump.lower()} in pump and {accum.lower()} \
in accumulator for this cycle.

SENSOR EVIDENCE:
Motor power and flow sensors indicate deviation from healthy baseline consistent \
with the detected fault condition. System efficiency readings show abnormal \
operating pattern — elevated variability suggests instability within the cycle.

CONSEQUENCE OF DEFERRAL:
{consequence}

RECOMMENDED ACTION:
{urgency} Inspect pump seals and accumulator charge pressure. \
Schedule maintenance work order referencing ISO 4413 section 5.4."""


if __name__ == "__main__":
    # Test with two cycles that have the same fault class but different
    # escalation probabilities — recommendations should differ

    base_shap = {
        'top_contributors': [
            {
                'feature': 'SE_mean',
                'description': 'system efficiency (average)',
                'shap_value': 0.071,
                'actual_value': 0.31,
                'direction': 'increases risk',
                'magnitude': 'high'
            },
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

    # Cycle A — low escalation risk
    cycle_a = {
        'cycle': 1,
        'pump_prediction': 'Weak leakage',
        'pump_probabilities': {
            'No leakage': 0.62,
            'Weak leakage': 0.33,
            'Severe leakage': 0.05   # low severe probability
        },
        'accumulator_prediction': 'Optimal pressure',
        'accumulator_probabilities': {
            'Optimal pressure': 0.91,
            'Slightly reduced pressure': 0.07,
            'Severely reduced pressure': 0.01,
            'Close to failure': 0.01
        },
        'requires_attention': True
    }

    # Cycle B — high escalation risk, same fault class
    cycle_b = {
        'cycle': 2,
        'pump_prediction': 'Weak leakage',
        'pump_probabilities': {
            'No leakage': 0.18,
            'Weak leakage': 0.54,
            'Severe leakage': 0.28   # high severe probability
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

    print("=" * 60)
    print("CYCLE A — Low escalation risk (severe prob: 5%)")
    print("=" * 60)
    print(generate_narrative(cycle_a, base_shap))

    print("\n" + "=" * 60)
    print("CYCLE B — High escalation risk (severe prob: 28%)")
    print("=" * 60)
    print(generate_narrative(cycle_b, base_shap))