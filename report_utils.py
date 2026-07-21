"""
Helper functions for PredictAFR: narrative text generation, rule-based
clinical recommendations, and PDF report export.

Kept separate from streamlit_app.py so the main app file stays focused on
layout/UI.
"""

from datetime import date

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def generate_narrative(patient: pd.Series, top_features: pd.DataFrame, risk_score: float) -> str:
    """Build a short free-text summary of the risk drivers for one patient."""
    drivers = ", ".join(
        f"{row.Feature} ({row.SHAP_Value:+.1%})" for row in top_features.itertuples()
    )
    af_type = patient["AF_Type"].lower()
    return (
        f"Patient {int(patient['PatientID'])} ({int(patient['Age'])}yo {patient['Sex'].lower()}) "
        f"has an estimated {risk_score:.0%} risk of AF recurrence within 12 months of "
        f"{patient['Ablation_Type'].lower()} ablation, presenting with {af_type} AF of "
        f"{int(patient['AF_Duration_Months'])} months' duration. The model's leading contributors "
        f"to this estimate are: {drivers}. LA volume index is {patient['LAVi']:.1f} mL/m2 and LVEF "
        f"is {patient['LVEF']:.0f}%, with an LA reservoir strain of {patient['LA_Reservoir_Strain']:.1f}%."
    )


def generate_recommendations(patient: pd.Series, risk_score: float) -> list[str]:
    """Simple rule-based recommendation list, for teaching illustration only."""
    recs = []

    if patient["AF_Type"] in ("Persistent", "Long-standing persistent"):
        recs.append("Consider a rhythm-control strategy given persistent/long-standing AF.")
    if risk_score >= 0.6:
        recs.append("Schedule an early 3-month follow-up given the elevated recurrence risk.")
    else:
        recs.append("Routine follow-up per standard post-ablation pathway.")
    if patient["Antiarrhythmic_Use"] == 0 and risk_score >= 0.5:
        recs.append("Consider commencing or optimising antiarrhythmic drug (AAD) therapy.")
    if patient["CHA2DS2_VASc"] >= 2 and patient["OAC_Use"] == 0:
        recs.append(
            f"Reassess anticoagulation: CHA2DS2-VASc is {int(patient['CHA2DS2_VASc'])} "
            "but the patient is not currently on an oral anticoagulant."
        )
    if patient["OSA"] == 1:
        recs.append("Refer for sleep study/CPAP optimisation - OSA is a modifiable recurrence risk factor.")
    if patient["BMI"] >= 30:
        recs.append(f"Weight management referral (BMI {patient['BMI']:.1f}) to reduce recurrence risk.")
    recs.append("Monitor for symptomatic recurrence and quality-of-life changes at each review.")

    return recs


class PatientReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(
            0, 10, "PredictAFR: AF Recurrence Risk Report", border=1,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
        )
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0, 10, f"Generated {date.today()} - Synthetic demo data - Not for clinical use",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
        )

    def section_title(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(238, 241, 246)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(1)

    def field_row(self, label, value):
        self.set_font("Helvetica", "", 10)
        self.cell(70, 6, label)
        self.cell(0, 6, str(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


FIELD_GROUPS = {
    "Demographics": [
        ("Age", "Age", "years"), ("Sex", "Sex", ""), ("BMI", "BMI", "kg/m2"),
        ("Alcohol_Use", "Alcohol use", ""), ("Smoking_Status", "Smoking status", ""),
        ("Drug_Use", "Recreational drug use", ""), ("APPLE_Score", "APPLE score", "/5"),
        ("CHA2DS2_VASc", "CHA2DS2-VASc", ""), ("HAS_BLED", "HAS-BLED", ""),
    ],
    "Comorbidities": [
        ("CHF", "Congestive heart failure", ""), ("Hypertension", "Hypertension", ""),
        ("Vascular_Disease", "Vascular disease", ""), ("Thyroid_Disorder", "Thyroid disorder", ""),
        ("Renal_Disorder", "Renal disorder", ""), ("Hepatic_Disorder", "Hepatic disorder", ""),
        ("Respiratory_Disorder", "Respiratory disorder", ""), ("OSA", "Obstructive sleep apnoea", ""),
        ("Diabetes", "Diabetes mellitus", ""), ("Prior_Stroke_TIA_TE", "Prior stroke/TIA/TE", ""),
        ("Prior_MI", "Prior myocardial infarction", ""),
    ],
    "Medications": [
        ("OAC_Use", "Oral anticoagulant", ""), ("Antiplatelet_Use", "Antiplatelet", ""),
        ("Antiarrhythmic_Use", "Antiarrhythmic", ""), ("Beta_Blocker_Use", "Beta blocker", ""),
        ("Statin_Use", "Statin", ""),
    ],
    "Biochemistry": [
        ("Haemoglobin", "Haemoglobin", "g/L"), ("Creatinine", "Creatinine", "umol/L"),
        ("eGFR_Over_90", "eGFR > 90", ""), ("ALT", "ALT", "U/L"),
    ],
    "AF Details": [
        ("AF_Type", "AF type", ""), ("AF_Duration_Months", "AF duration", "months"),
        ("Concomitant_Flutter", "Concomitant atrial flutter", ""),
        ("Prior_Cardioversion", "Prior DC cardioversion", ""), ("Ablation_Type", "Ablation type", ""),
    ],
    "Imaging": [
        ("IVSd", "IVSd", "cm"), ("LVVi", "LV volume index", "mL/m2"), ("LVEF", "LVEF", "%"),
        ("LAVi", "LA volume index", "mL/m2"), ("E_A_Ratio", "E/A ratio", ""),
        ("E_e_prime_avg", "E/e' average", ""),
    ],
    "LA Strain": [
        ("Time_to_Peak_ms", "Time to peak", "ms"), ("LA_Reservoir_Strain", "LA reservoir strain", "%"),
        ("LA_Conduit_Strain", "LA conduit strain", "%"), ("LA_Contractile_Strain", "LA contractile strain", "%"),
    ],
}


BINARY_COLUMNS = {
    "CHF", "Hypertension", "Vascular_Disease", "Thyroid_Disorder", "Renal_Disorder",
    "Hepatic_Disorder", "Respiratory_Disorder", "OSA", "Diabetes", "Prior_Stroke_TIA_TE",
    "Prior_MI", "OAC_Use", "Antiplatelet_Use", "Antiarrhythmic_Use", "Beta_Blocker_Use",
    "Statin_Use", "eGFR_Over_90", "Concomitant_Flutter", "Prior_Cardioversion",
}


def _format_value(patient, col, unit) -> str:
    val = patient[col]
    if col in BINARY_COLUMNS:
        return "Yes" if val == 1 else "No"
    if unit:
        return f"{val} {unit}"
    return str(val)


def build_pdf(patient: pd.Series, risk_score: float, narrative: str, recommendations: list[str]) -> bytes:
    pdf = PatientReportPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Patient ID: {int(patient['PatientID'])}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0, 10, f"Calculated 12-Month Recurrence Risk: {risk_score:.0%}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True,
    )
    pdf.ln(3)

    for group_name, fields in FIELD_GROUPS.items():
        pdf.section_title(group_name)
        for col, label, unit in fields:
            pdf.field_row(label, _format_value(patient, col, unit))
        pdf.ln(2)

    pdf.section_title("Clinical Narrative")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, narrative, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    pdf.section_title("Clinical Recommendations")
    pdf.set_font("Helvetica", "", 10)
    for i, rec in enumerate(recommendations, start=1):
        pdf.multi_cell(0, 6, f"{i}. {rec}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    output = pdf.output()
    return bytes(output)
