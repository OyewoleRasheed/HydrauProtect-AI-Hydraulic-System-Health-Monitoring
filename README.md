# 🔧 HydrauProtect AI — Hydraulic System Health Monitoring

> An end-to-end AI system that analyses hydraulic system sensor data, diagnoses pump leakage and accumulator pressure faults, explains *why* using SHAP, and generates plain-English maintenance reports for shift engineers — automatically.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Railway-46E3B7)](https://hydrauprotectai.up.railway.app)
---

## 🔍 What Problem Does This Solve?

Hydraulic system failures in oil and gas facilities are expensive, dangerous, and often preventable. Traditional monitoring requires engineers to manually review sensor logs and cross-reference fault thresholds — a slow, reactive process.

**HydrauProtect AI makes this proactive and automatic:**

- Upload a CSV of sensor readings from your SCADA system
- Get instant diagnosis of pump leakage and accumulator pressure state
- Understand *which sensors* drove the finding — in plain engineering language
- Download a structured PDF maintenance report ready for your work order system

---

## 🧠 How It Works

```
Engineer uploads CSV (SCADA sensor export)
              │
              ▼
   Feature Engineering (81 features per cycle)
              │
              ▼
   ┌──────────────────────────┐
   │  Pump Leakage Model      │  →  No leakage / Weak leakage / Severe leakage
   │  Accumulator Model       │  →  Optimal / Slightly reduced / Severely reduced / Close to failure
   └──────────────────────────┘
              │
              ▼  (flagged cycles only)
   SHAP Explainer  →  Top 5 contributing sensors + direction + magnitude
              │
              ▼
   Groq LLM (Llama 3.1 8B)  →  Structured maintenance finding in plain English
              │
              ▼
   JSON API response  +  Downloadable PDF Report
```

---

## 📊 What Gets Diagnosed

### Pump Leakage (3 classes)
| Prediction | Meaning |
|---|---|
| No leakage | System healthy |
| Weak leakage | Early warning — monitor closely |
| Severe leakage | Immediate intervention required |

### Accumulator Pressure (4 classes)
| Prediction | Meaning |
|---|---|
| Optimal pressure (130 bar) | System healthy |
| Slightly reduced pressure (115 bar) | Minor deviation — schedule check |
| Severely reduced pressure (100 bar) | Significant fault — act soon |
| Close to failure (90 bar) | Critical — stop and inspect |

---

## 🛢️ Sensor Coverage

The system processes 16 hydraulic sensors, each summarised into 5 statistics (mean, std, min, max, range) per operating cycle — **81 features total**:

| Sensor | Description |
|---|---|
| PS1–PS6 | System pressure sensors + return line pressure |
| EPS1 | Motor power |
| FS1–FS2 | Volume flow sensors |
| TS1–TS4 | Temperature sensors |
| VS1 | Vibration |
| SE / CE | System efficiency / Cooling efficiency |

---

## 📋 Sample Output

**JSON response from `/analyse`:**
```json
{
  "summary": {
    "total_cycles": 10,
    "flagged_cycles": 3,
    "healthy_cycles": 7,
    "action_required": true
  },
  "predictions": [
    {
      "cycle": 2,
      "pump_prediction": "Weak leakage",
      "pump_probabilities": {"No leakage": 0.18, "Weak leakage": 0.71, "Severe leakage": 0.11},
      "accumulator_prediction": "Slightly reduced pressure",
      "requires_attention": true
    }
  ],
  "narratives": {
    "2": "FINDING: Weak pump leakage detected...\nSENSOR EVIDENCE: ...\nRECOMMENDED ACTION: Address within 24 hours..."
  }
}
```

**LLM-generated maintenance finding (structured for field engineers):**
```
FINDING:
Weak leakage detected in hydraulic pump — medium severity.

SENSOR EVIDENCE:
Motor power readings are elevated above healthy baseline, and volume flow
sensors indicate reduced throughput consistent with internal bypass leakage.
Return line pressure shows minor deviation from expected range.

CONSEQUENCE OF DEFERRAL:
Unaddressed weak leakage progresses to severe leakage, risking unplanned
shutdown and potential seal failure.

RECOMMENDED ACTION:
Address within 24 hours. Inspect pump seals and check internal bypass valve.
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Models | Scikit-learn / XGBoost (Random Forest + Gradient Boosting) |
| Explainability | SHAP (TreeExplainer) |
| LLM Narrative | Groq API — Llama 3.1 8B Instant |
| Backend | Flask |
| PDF Reports | `FPDF` |
| Frontend | HTML/CSS/JS |
| Deployment | Railway |

---

## 📁 Project Structure

```
├── api/
│   └── app.py                  # Flask routes — /analyse, /report, /template, /health
├── src/
│   ├── feature_engineering.py  # CSV parsing, feature extraction, template generation
│   ├── predictor.py            # Loads models, runs inference, returns per-cycle results
│   ├── explainer.py            # SHAP TreeExplainer — top 5 sensor contributors
│   ├── llm_narrator.py         # Groq API call — structured plain-English maintenance report
│   └── report_generator.py    # PDF report generation
├── models/
│   ├── pump_model.pkl          # Trained pump leakage classifier
│   ├── accumulator_model.pkl   # Trained accumulator pressure classifier
│   └── feature_names.pkl       # Saved feature column order
├── frontend/
│   └── index.html              # Engineer-facing UI
├── notebooks/
│   └── explore.ipynb           # EDA and model training notebook
├── engineer_upload_template.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env

# Run the app
python api/app.py
```

App runs at `http://localhost:5000`

---

## 🔑 Environment Variables

```env
GROQ_API_KEY=your_groq_api_key_here
```

> If `GROQ_API_KEY` is not set, the system automatically falls back to a rule-based report generator — the app never crashes due to a missing API key.

---

## 📡 API Reference

### `POST /analyse`
Upload a CSV of sensor readings. Returns predictions, SHAP explanations, and LLM narratives for all flagged cycles.

**Request:** `multipart/form-data` with `file` field (CSV, max 16MB)

**Response:** JSON with `summary`, `predictions`, `narratives`, `shap_explanations`

---

### `POST /report`
Same CSV upload as `/analyse` — returns a downloadable PDF maintenance report.

**Response:** `application/pdf` — `hydrauprotect_maintenance_report.pdf`

---

### `GET /template`
Download the empty CSV template (`hydrauprotect_upload_template.csv`) to fill in with SCADA data.

---

### `GET /health`
```json
{ "status": "running", "service": "HydrauProtect AI", "version": "1.0.0" }
```

---

## 💡 Key Design Decisions

- **Statistics per cycle, not raw time-series** — Each operating cycle is summarised into 81 statistical features, making inference fast and the input format simple enough for a field engineer to fill in manually from a SCADA export.
- **SHAP only on flagged cycles** — Running SHAP on every cycle is expensive. The system only explains cycles where `requires_attention = True`, keeping response times fast for healthy systems.
- **LLM grounding** — The prompt explicitly anchors the LLM to the ML model's confirmed diagnosis, preventing the language model from softening or contradicting the fault severity.
- **Graceful LLM fallback** — If the Groq API is unavailable, a rule-based severity-to-urgency mapping generates a usable report automatically. The system never returns an empty response.
- **Plain language for engineers** — SHAP values and model probabilities are deliberately hidden from the final report. Engineers see sensor readings and physical interpretations, not data science terminology.

---

## 🗺️ Roadmap

- [ ] Real-time streaming mode — analyse cycles as they arrive from SCADA
- [ ] Accumulator SHAP explanation (currently pump-only)
- [ ] Trend analysis across multiple uploads — detect gradual degradation
- [ ] Integration with maintenance work order systems (SAP PM / Maximo)

---

## 📊 Dataset

Models trained on the **UCI Hydraulic System Condition Monitoring Dataset** — 2205 operating cycles across 16 sensors at 1–100 Hz sampling rates.

> [UCI ML Repository — Condition monitoring of hydraulic systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)

---

## 👤 Author

**[Rasheed Oyewole]**  
AI/ML Engineer 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/rasheed-adebayo-oyewole-gmnse/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/OyewoleRasheed)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ **Disclaimer:** HydrauProtect AI is a decision-support tool. All maintenance decisions must be reviewed and approved by qualified engineers. Do not use as a sole basis for safety-critical actions.