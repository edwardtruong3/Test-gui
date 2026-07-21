"""
PredictAFR — Clinical Decision Support Demo
Machine-learning-style prediction of atrial fibrillation recurrence
12 months after catheter ablation.

IMPORTANT: All patient data is synthetic (see generate_data.py) and the
"model" is a hand-built mock explainer for teaching purposes. This app is a
research/education demo only and must not be used for clinical decisions.
"""

from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from report_utils import build_pdf, generate_narrative, generate_recommendations

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="PredictAFR", page_icon="🫀", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #fbfcfe; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #eef1f6;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stExpander"] {
        border: 1px solid #eef1f6;
        border-radius: 10px;
        background-color: #ffffff;
    }
    .section-caption { color: #667085; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

NUMERIC_COMPARISON_FEATURES = ["Age", "LAVi", "AF_Duration_Months", "LA_Reservoir_Strain", "LVEF"]

FIELD_GROUPS = {
    "Demographics": [
        ("Age", "Age", "years"), ("Sex", "Sex", ""), ("BMI", "BMI", "kg/m²"),
        ("Alcohol_Use", "Alcohol use", ""), ("Smoking_Status", "Smoking status", ""),
        ("Drug_Use", "Recreational drug use", ""), ("APPLE_Score", "APPLE score", "/5"),
        ("CHA2DS2_VASc", "CHA₂DS₂-VASc", ""), ("HAS_BLED", "HAS-BLED", ""),
    ],
    "Comorbidities & Prior Events": [
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
        ("Haemoglobin", "Haemoglobin", "g/L"), ("Creatinine", "Creatinine", "μmol/L"),
        ("eGFR_Over_90", "eGFR > 90", ""), ("ALT", "ALT", "U/L"),
    ],
    "AF Details": [
        ("AF_Type", "AF type", ""), ("AF_Duration_Months", "AF duration", "months"),
        ("Concomitant_Flutter", "Concomitant atrial flutter", ""),
        ("Prior_Cardioversion", "Prior DC cardioversion", ""), ("Ablation_Type", "Ablation type", ""),
    ],
    "Imaging": [
        ("IVSd", "IVSd", "cm"), ("LVVi", "LV volume index", "mL/m²"), ("LVEF", "LVEF", "%"),
        ("LAVi", "LA volume index", "mL/m²"), ("E_A_Ratio", "E/A ratio", ""),
        ("E_e_prime_avg", "E/e' average", ""),
    ],
    "Left Atrial Strain": [
        ("Time_to_Peak_ms", "Time to peak", "ms"), ("LA_Reservoir_Strain", "LA reservoir strain", "%"),
        ("LA_Conduit_Strain", "LA conduit strain", "%"), ("LA_Contractile_Strain", "LA contractile strain", "%"),
    ],
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    patients = pd.read_csv("patients.csv")
    shap_values = pd.read_csv("shap_values.csv")
    return patients, shap_values


def format_value(row, col, unit) -> str:
    val = row[col]
    if col in BINARY_COLUMNS:
        return "Yes" if val == 1 else "No"
    if unit:
        return f"{val} {unit}"
    return str(val)


def nearest_neighbours(patients: pd.DataFrame, patient_id: int, k: int = 5) -> pd.DataFrame:
    """Standardised Euclidean distance over a small clinically-relevant feature set."""
    features = patients[NUMERIC_COMPARISON_FEATURES]
    z = (features - features.mean()) / features.std(ddof=0)
    target = z.loc[patients["PatientID"] == patient_id].iloc[0]
    distances = np.sqrt(((z - target) ** 2).sum(axis=1))
    ranked = patients.assign(_distance=distances.values)
    ranked = ranked[ranked["PatientID"] != patient_id].sort_values("_distance")
    return ranked.head(k)


def render_waterfall(shap_row: pd.DataFrame, patient_id: int):
    chart_data = shap_row.copy()
    bars = (
        alt.Chart(chart_data)
        .mark_bar(height=20, cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "SHAP_Value:Q",
                title="Attribution impact on risk score",
                axis=alt.Axis(format="+.0%"),
            ),
            y=alt.Y(
                "Feature:N",
                title=None,
                sort=alt.EncodingSortField(field="SHAP_Value", order="descending"),
            ),
            color=alt.Color(
                "Type:N",
                legend=alt.Legend(title="Factor", orient="top-left"),
                scale=alt.Scale(
                    domain=["Risk Factor", "Protective Factor"],
                    range=["#e45756", "#4c78a8"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("SHAP_Value:Q", format="+.1%", title="Contribution"),
                alt.Tooltip("Type:N"),
            ],
        )
        .properties(
            height=380,
            title=alt.TitleParams(
                text=f"Feature attribution — Patient {patient_id}",
                subtitle="Left = protective | Right = elevates risk",
                anchor="start",
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#666",
            ),
        )
    )
    zero_rule = alt.Chart(pd.DataFrame([{"zero": 0}])).mark_rule(color="#333", strokeWidth=1.5).encode(x="zero:Q")
    return alt.layer(bars, zero_rule).configure_view(strokeWidth=0)


# --------------------------------------------------------------------------
# Load data & establish binary columns (for Yes/No formatting)
# --------------------------------------------------------------------------
patients_df, shap_df = load_data()
BINARY_COLUMNS = {
    "CHF", "Hypertension", "Vascular_Disease", "Thyroid_Disorder", "Renal_Disorder",
    "Hepatic_Disorder", "Respiratory_Disorder", "OSA", "Diabetes", "Prior_Stroke_TIA_TE",
    "Prior_MI", "OAC_Use", "Antiplatelet_Use", "Antiarrhythmic_Use", "Beta_Blocker_Use",
    "Statin_Use", "eGFR_Over_90", "Concomitant_Flutter", "Prior_Cardioversion",
}

# --------------------------------------------------------------------------
# Sidebar: patient selection
# --------------------------------------------------------------------------
st.sidebar.header("🏥 Patient Selection")
selected_id = st.sidebar.selectbox("Select Patient ID", patients_df["PatientID"])
patient = patients_df.loc[patients_df["PatientID"] == selected_id].iloc[0]
risk_score = float(patient["Risk_Score"])
risk_level = "High Risk" if risk_score > 0.6 else "Moderate Risk" if risk_score > 0.3 else "Low Risk"

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Summary**")
st.sidebar.write(f"Age {int(patient['Age'])} · {patient['Sex']}")
st.sidebar.write(f"{patient['AF_Type']} AF, {int(patient['AF_Duration_Months'])} months")
st.sidebar.write(f"LAVi {patient['LAVi']:.1f} mL/m² · LVEF {patient['LVEF']:.0f}%")

st.sidebar.markdown("---")
st.sidebar.header("💬 AI Assistant")
st.sidebar.info("Chatbot functionality is not available in this interactive demo.")
st.sidebar.chat_input("Ask about this patient...", disabled=True)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🫀 PredictAFR")
st.markdown(
    "##### Machine learning-style prediction of atrial fibrillation recurrence "
    "12 months after catheter ablation"
)

tab_risk, tab_profile, tab_cases, tab_export = st.tabs(
    ["📈 Risk Assessment", "📋 Comprehensive Profile", "📊 Similar Cases", "📄 Export Report"]
)

# --------------------------------------------------------------------------
# Tab 1: Risk Assessment
# --------------------------------------------------------------------------
with tab_risk:
    patient_shap = shap_df[shap_df["PatientID"] == selected_id].sort_values(
        "SHAP_Value", ascending=False
    )
    narrative = generate_narrative(patient, patient_shap.head(3), risk_score)
    recommendations = generate_recommendations(patient, risk_score)

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        st.metric("12-Month Recurrence Risk", f"{risk_score:.0%}", delta=risk_level, delta_color="inverse")
    with col2:
        st.metric(
            "LA Volume Index",
            f"{patient['LAVi']:.1f} mL/m²",
            delta="Above threshold" if patient["LAVi"] > 40 else "Normal",
        )
    with col3:
        st.markdown("##### 🤖 Model Summary")
        st.info(narrative)

    st.divider()
    st.subheader("🔍 Decision Logic (Feature Attribution)")
    st.altair_chart(render_waterfall(patient_shap, selected_id), width='stretch')

    st.subheader("Clinical Recommendations")
    for rec in recommendations:
        st.markdown(f"- {rec}")

# --------------------------------------------------------------------------
# Tab 2: Comprehensive Profile
# --------------------------------------------------------------------------
with tab_profile:
    st.markdown(f"#### Patient {int(patient['PatientID'])} — Comprehensive Record")
    group_names = list(FIELD_GROUPS.keys())
    profile_tabs = st.tabs(group_names)
    for tab, group in zip(profile_tabs, group_names):
        with tab:
            fields = FIELD_GROUPS[group]
            rows = [
                {"Attribute": label, "Value": format_value(patient, col, unit)}
                for col, label, unit in fields
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

# --------------------------------------------------------------------------
# Tab 3: Case-Based Reasoning
# --------------------------------------------------------------------------
with tab_cases:
    st.subheader("📊 Case-Based Reasoning")
    st.markdown(
        "<span class='section-caption'>Nearest neighbours by standardised Euclidean distance "
        f"over {', '.join(NUMERIC_COMPARISON_FEATURES)}.</span>",
        unsafe_allow_html=True,
    )

    neighbours = nearest_neighbours(patients_df, selected_id, k=5)

    current_display = patient.to_frame().T.copy()
    current_display["Category"] = "⭐ Current patient"
    neighbours_display = neighbours.copy()
    neighbours_display["Category"] = "📁 Historical match"

    display_cols = ["Category", "PatientID", "Age", "AF_Type", "LAVi", "LVEF",
                     "LA_Reservoir_Strain", "AF_Duration_Months", "Outcome"]
    comparison_df = pd.concat([current_display, neighbours_display], axis=0)[display_cols]

    def highlight_current(row):
        if row["Category"] == "⭐ Current patient":
            return ["background-color: #1f77b4; color: white; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        comparison_df.style.apply(highlight_current, axis=1),
        hide_index=True,
        width='stretch',
    )

    recurred_count = (neighbours_display["Outcome"] == "Recurred").sum()
    st.write(
        f"**Historical pattern:** {recurred_count} of 5 matched cases experienced recurrence "
        f"({recurred_count / 5:.0%})."
    )

# --------------------------------------------------------------------------
# Tab 4: Export Report
# --------------------------------------------------------------------------
with tab_export:
    st.subheader("📄 Export Prediction Report")
    st.write(
        "Generate an archival PDF report covering the comprehensive patient record, "
        "feature attribution summary, and current recommendations."
    )

    patient_shap = shap_df[shap_df["PatientID"] == selected_id].sort_values(
        "SHAP_Value", ascending=False
    )
    narrative = generate_narrative(patient, patient_shap.head(3), risk_score)
    recommendations = generate_recommendations(patient, risk_score)

    try:
        pdf_bytes = build_pdf(patient, risk_score, narrative, recommendations)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"AF_Report_Patient_{selected_id}_{date.today()}.pdf",
            mime="application/pdf",
            width='stretch',
        )
        st.success("✅ Report compiled cleanly")
    except Exception as exc:
        st.error(f"❌ Document compilation exception: {exc}")

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool uses synthetic data and a mock model for research/education "
    "purposes only. It should not replace clinical judgement or be used as the sole basis for "
    "treatment decisions."
)
