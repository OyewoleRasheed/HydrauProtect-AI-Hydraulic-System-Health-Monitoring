
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_file,render_template
from werkzeug.utils import secure_filename
import pandas as pd
import tempfile
import traceback

from src.feature_engineering import extract_features_from_csv, generate_csv_template
from src.predictor import predict
from src.explainer import BASE_DIR, explain_prediction
from src.llm_narrator import generate_narrative
from src.report_generator import generate_report

app = Flask(__name__,
    template_folder=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'frontend'
    )
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'service': 'HydrauProtect AI',
        'version': '1.0.0'
    })


@app.route('/template', methods=['GET'])
def download_template():
    """
    Engineer downloads this CSV template,
    fills it in with their SCADA export,
    and uploads it to /analyse
    """
    template = generate_csv_template()
    
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.csv',
        delete=False,
        prefix='hydrauprotect_template_'
    ) as f:
        template.to_csv(f, index=False)
        temp_path = f.name

    return send_file(
        temp_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='hydrauprotect_upload_template.csv'
    )

@app.route('/samples/<path:filename>')
def serve_sample(filename):
    samples_dir = os.path.join(BASE_DIR, 'samples')
    return send_file(os.path.join(samples_dir, filename))

@app.route('/analyse', methods=['POST'])
def analyse():
    """
    Main endpoint. Engineer uploads CSV, gets back JSON results.
    
    Request: multipart/form-data with 'file' field
    Response: JSON with predictions, explanations, narratives
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are accepted'}), 400

    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix='.csv',
            delete=False
        ) as f:
            file.save(f)
            temp_path = f.name

        features_df = extract_features_from_csv(temp_path)
        print(f"Features extracted: {features_df.shape}")

        predictions = predict(features_df)
        print(f"Predictions complete: {len(predictions)} cycles")

        flagged = [p for p in predictions if p['requires_attention']]
        print(f"Cycles requiring attention: {len(flagged)}")

        narratives = {}
        shap_explanations = {}

        for result in flagged:
            cycle_idx = result['cycle'] - 1
            cycle_features = features_df.iloc[[cycle_idx]]

            pump_class_map = {
                'No leakage': 0,
                'Weak leakage': 1,
                'Severe leakage': 2
            }
            pump_class_idx = pump_class_map.get(
                result['pump_prediction'], 0
            )

            shap_exp = explain_prediction(
                cycle_features,
                pump_class_idx,
                model_type='pump'
            )
            shap_explanations[result['cycle']] = shap_exp

            narrative = generate_narrative(result, shap_exp)
            narratives[result['cycle']] = narrative

        os.unlink(temp_path)

        return jsonify({
            'status': 'success',
            'summary': {
                'total_cycles': len(predictions),
                'flagged_cycles': len(flagged),
                'healthy_cycles': len(predictions) - len(flagged),
                'action_required': len(flagged) > 0
            },
            'predictions': predictions,
            'narratives': narratives,
            'shap_explanations': {
                k: {
                    'top_contributors': v['top_contributors']
                }
                for k, v in shap_explanations.items()
            }
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 422

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Analysis failed', 'detail': str(e)}), 500


@app.route('/report', methods=['POST'])
def generate_pdf_report():
    """
    Takes JSON results from /analyse and generates a PDF report.
    
    Request: multipart/form-data with 'file' field
    Response: PDF file download
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are accepted'}), 400

    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix='.csv',
            delete=False
        ) as f:
            file.save(f)
            temp_path = f.name

        features_df = extract_features_from_csv(temp_path)
        predictions = predict(features_df)
        flagged = [p for p in predictions if p['requires_attention']]

        narratives = {}
        shap_explanations = {}

        for result in flagged:
            cycle_idx = result['cycle'] - 1
            cycle_features = features_df.iloc[[cycle_idx]]

            pump_class_map = {
                'No leakage': 0,
                'Weak leakage': 1,
                'Severe leakage': 2
            }
            pump_class_idx = pump_class_map.get(
                result['pump_prediction'], 0
            )

            shap_exp = explain_prediction(
                cycle_features,
                pump_class_idx,
                model_type='pump'
            )
            shap_explanations[result['cycle']] = shap_exp

            narrative = generate_narrative(result, shap_exp)
            narratives[result['cycle']] = narrative
            
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(BASE_DIR, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        report_path = generate_report(
            predictions,
            narratives,
            shap_explanations,
            output_path=os.path.join(reports_dir, 'hydrauprotect_report.pdf')
)

        # os.makedirs('reports', exist_ok=True)
        # report_path = generate_report(
        #     predictions,
        #     narratives,
        #     shap_explanations,
        #     output_path=f'reports/hydrauprotect_report.pdf'
        # )

        os.unlink(temp_path)

        return send_file(
            report_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='hydrauprotect_maintenance_report.pdf'
        )

    except ValueError as e:
        return jsonify({'error': str(e)}), 422

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Report generation failed', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)