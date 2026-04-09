from fpdf import FPDF
from datetime import datetime
import os

class MaintenanceReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, 'Hydraulic System Maintenance Report', ln=True, align='C')
        self.set_font('Arial', '', 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                  ln=True, align='C')
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, 
                  'HydrauProtect AI - Hydraulic System Maintenance Decision Support', 
                  align='C')

    def section_title(self, title: str):
        self.set_font('Arial', 'B', 11)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(245, 245, 245)
        self.cell(0, 8, f'  {title}', ln=True, fill=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font('Arial', '', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(2)

  
    def status_badge(self, label: str, value: str, severity: str):
        colors = {
            'low': (39, 174, 96),
            'medium': (230, 126, 34),
            'high': (192, 57, 43),
            'critical': (123, 36, 28)
        }
        color = colors.get(severity, (100, 100, 100))

        self.set_font('Arial', '', 10)
        self.set_text_color(80, 80, 80)
        self.cell(60, 8, label, ln=False)

        self.set_font('Arial', 'B', 10)
        self.set_text_color(*color)
        self.cell(120, 8, value, ln=True)  # explicit width
        self.set_text_color(50, 50, 50)
            

    def probability_bar(self, label: str, probability: float):
        bar_width = 80
        filled = int(bar_width * probability)
        
        self.set_font('Arial', '', 9)
        self.set_text_color(80, 80, 80)
        self.cell(55, 6, label, ln=False)
        
        x = self.get_x()
        y = self.get_y()
        
        self.set_fill_color(220, 220, 220)
        self.rect(x, y+1, bar_width, 4, 'F')
        
        if probability > 0.6:
            self.set_fill_color(192, 57, 43)
        elif probability > 0.3:
            self.set_fill_color(230, 126, 34)
        else:
            self.set_fill_color(39, 174, 96)
            
        self.rect(x, y+1, filled, 4, 'F')
        
        self.set_xy(x + bar_width + 3, y)
        self.set_font('Arial', 'B', 9)
        self.cell(20, 6, f'{probability*100:.0f}%', ln=True)

pdf = MaintenanceReport()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_margins(10, 20, 10)


def generate_report(
    prediction_results: list,
    narratives: dict,
    shap_explanations: dict,
    output_path: str = None
) -> str:
    """
    Generates a PDF maintenance report.
    
    prediction_results: list of cycle predictions
    narratives: dict mapping cycle number to narrative text
    shap_explanations: dict mapping cycle number to SHAP explanation
    output_path: where to save PDF
    
    Returns path to saved PDF.
    """
    # pdf = MaintenanceReport()
    # pdf.set_auto_page_break(auto=True, margin=15)
    # pdf.add_page()

    flagged = [r for r in prediction_results if r['requires_attention']]
    healthy = [r for r in prediction_results if not r['requires_attention']]

    pdf.section_title('Executive Summary')
    pdf.body_text(
        f"Total cycles analysed: {len(prediction_results)}\n"
        f"Cycles requiring attention: {len(flagged)}\n"
        f"Healthy cycles: {len(healthy)}\n"
        f"Overall system status: {'ACTION REQUIRED' if flagged else 'NORMAL'}"
    )

    if flagged:
        pdf.section_title('System Health Overview')
        
        pump_severity_map = {
            'No leakage': 'low',
            'Weak leakage': 'medium',
            'Severe leakage': 'high'
        }
        accum_severity_map = {
            'Optimal pressure': 'low',
            'Slightly reduced pressure': 'medium',
            'Severely reduced pressure': 'high',
            'Close to failure': 'critical'
        }

        worst_pump = max(flagged, 
                        key=lambda x: ['No leakage', 'Weak leakage', 
                                      'Severe leakage'].index(x['pump_prediction']))
        # With this
        accum_severity_order = [
            'Optimal pressure',
            'Slightly reduced pressure', 
            'Severely reduced pressure',
            'Close to failure'
        ]

        worst_accum = max(
            flagged,
            key=lambda x: accum_severity_order.index(x['accumulator_prediction'])
            if x['accumulator_prediction'] in accum_severity_order else 0
        )

        pdf.status_badge(
            'Pump condition:', 
            worst_pump['pump_prediction'],
            pump_severity_map[worst_pump['pump_prediction']]
        )
        pdf.status_badge(
            'Accumulator condition:', 
            worst_accum['accumulator_prediction'],
            accum_severity_map[worst_accum['accumulator_prediction']]
        )
        
        pdf.ln(4)

    for result in flagged:
        cycle_num = result['cycle']
        
        pdf.section_title(f'Cycle {cycle_num} - Detailed Finding')

        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 6, 'Pump Leakage Probabilities:', ln=True)
        pdf.ln(1)
        
        for label, prob in result['pump_probabilities'].items():
            pdf.probability_bar(label, prob)
        pdf.ln(3)

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'Accumulator Condition Probabilities:', ln=True)
        pdf.ln(1)
        
        for label, prob in result['accumulator_probabilities'].items():
            pdf.probability_bar(label, prob)
        pdf.ln(3)

        if cycle_num in shap_explanations:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, 'Top Contributing Sensors:', ln=True)
            pdf.ln(1)
            
            for contrib in shap_explanations[cycle_num]['top_contributors'][:3]:
                direction_symbol = '+' if contrib['direction'] == 'increases risk' else '-'
                pdf.set_font('Arial', '', 9)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(0, 6,
                    f"  [{direction_symbol}] {contrib['description'].title()} "
                    f"- {contrib['magnitude']} impact "
                    f"(value: {contrib['actual_value']})",
                    ln=True
                )
            pdf.ln(2)

        if cycle_num in narratives:
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 6, 'AI Maintenance Finding:', ln=True)
            pdf.ln(1)
            
            pdf.set_font('Arial', '', 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_fill_color(250, 250, 250)
            pdf.multi_cell(0, 6, narratives[cycle_num], fill=True)
            pdf.ln(4)

    pdf.section_title('Regulatory Reference')
    pdf.body_text(
        "ISO 4413 - Hydraulic fluid power: General rules and safety requirements\n"
        "API 686 - Recommended Practice for Machinery Installation and "
        "Installation Design\n"
        "ISO 18435 - Diagnostics and prognostics for hydraulic systems\n\n"
        "All maintenance actions should be documented in the facility CMMS "
        "and approved by the maintenance superintendent before execution."
    )

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/maintenance_report_{timestamp}.pdf"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    
    return output_path


if __name__ == "__main__":
    test_predictions = [
        {
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
        },
        {
            'cycle': 2,
            'pump_prediction': 'No leakage',
            'pump_probabilities': {
                'No leakage': 0.95,
                'Weak leakage': 0.04,
                'Severe leakage': 0.01
            },
            'accumulator_prediction': 'Optimal pressure',
            'accumulator_probabilities': {
                'Optimal pressure': 0.91,
                'Slightly reduced pressure': 0.07,
                'Severely reduced pressure': 0.02,
                'Close to failure': 0.00
            },
            'requires_attention': False
        }
    ]

    test_narratives = {
        1: """FINDING:
Weak internal pump leakage detected with slightly reduced accumulator pressure, indicating moderate system risk.

SENSOR EVIDENCE:
Motor power consumption is elevated above normal operating baseline, consistent with the pump working harder to compensate for internal fluid bypass. Volume flow readings are below expected values, confirming reduced pump delivery efficiency.

CONSEQUENCE OF DEFERRAL:
Continued operation without intervention will accelerate seal degradation, progressing to severe leakage and potential unplanned shutdown.

RECOMMENDED ACTION:
Within 24 hours - inspect pump shaft seals and check accumulator pre-charge pressure. Raise maintenance work order referencing ISO 4413 section 5.4."""
    }

    test_shap = {
        1: {
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
                },
                {
                    'feature': 'PS1_mean',
                    'description': 'system pressure 1 (average)',
                    'shap_value': 0.018,
                    'actual_value': 187.4,
                    'direction': 'increases risk',
                    'magnitude': 'low'
                }
            ]
        }
    }

    print("Generating test report...")
    path = generate_report(test_predictions, test_narratives, test_shap)
    print(f"Report saved to: {path}")
    print("Open the PDF and check it looks correct")