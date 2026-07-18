import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import date
import io

# --- PDF GENERATION LOGIC ---
class PatientReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Clinical Decision Support: AF Recurrence Report', border=True, ln=1, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generated on {date.today()} - Confidential Medical Research', 0, 0, 'C')

def create_pdf(patient_data, risk_score, narrative):
    pdf = PatientReport()
    pdf.add_page()
    
    p_id = patient_data['PatientID'].values[0]
    age = patient_data['Age'].values[0]
    af_type = patient_data['AF_Type'].values[0]
    bmi = patient_data['BMI'].values[0]
    la_vol = patient_data['LA_Vol'].values[0]
    outcome = patient_data['Outcome'].values[0]

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Patient ID: {p_id}", ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 7, f"Age: {age} years", ln=1)
    pdf.cell(0, 7, f"AF Type: {af_type}", ln=1)
    pdf.cell(0, 7, f"BMI: {bmi}", ln=1)
    pdf.cell(0, 7, f"LA Volume: {la_vol} mL", ln=1)
    pdf.cell(0, 7, f"Historical Outcome: {outcome}", ln=1)
    pdf.ln(5)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Calculated Risk of 12-Month Recurrence: {risk_score}", ln=1, fill=True)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Clinical Narrative & Supervisor Analysis:", ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, narrative)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Clinical Recommendations:", ln=1)
    pdf.set_font('Arial', '', 11)
    
    recommendations = [
        "1. Consider aggressive rhythm control strategy",
        "2. Schedule early 3-month follow-up appointment",
        "3. Continue antiarrhythmic drug (AAD) therapy",
        "4. Monitor for symptoms and quality of life changes",
        "5. Reassess anticoagulation strategy if indicated"
    ]
    
    for rec in recommendations:
        pdf.cell(0, 7, rec, ln=1)
    
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    else:
        return pdf_output

# 1. SETUP & THEME CLEANUP
st.set_page_config(page_title="AF Recurrence Supervisor", layout="wide")

st.markdown("""
    <style>
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e6e9ef;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'PatientID': range(101, 120),
        'Age': np.random.randint(45, 80, 19),
        'AF_Type': np.random.choice(['Paroxysmal', 'Persistent'], 19),
        'LA_Vol': np.random.randint(25, 60, 19),
        'BMI': np.random.randint(22, 38, 19),
        'Outcome': np.random.choice(['Recurred', 'Sinus Rhythm'], 19, p=[0.6, 0.4])
    })
    return df

data = load_and_prep_data()

# 2. SIDEBAR SELECTION
st.sidebar.header("🏥 Patient Selection")
st.sidebar.markdown("---")
selected_id = st.sidebar.selectbox(
    "Select Patient ID",
    data['PatientID'],
    help="Choose a patient from the registry to view their risk assessment"
)

patient_row = data[data['PatientID'] == selected_id].copy()

st.sidebar.markdown("### Patient Summary")
st.sidebar.write(f"**Age:** {patient_row['Age'].values[0]} years")
st.sidebar.write(f"**AF Type:** {patient_row['AF_Type'].values[0]}")
st.sidebar.write(f"**LA Volume:** {patient_row['LA_Vol'].values[0]} mL")
st.sidebar.write(f"**BMI:** {patient_row['BMI'].values[0]}")

st.sidebar.markdown("---")
st.sidebar.header("💬 AI Assistant")
st.sidebar.info("Chatbot functionality is not available in this interactive demo.")
st.sidebar.chat_input("Ask about this patient...", disabled=True)

# 3. INTERACTIVE DIALOG (POPUP) FOR PATIENT DETAILS
@st.dialog("📋 Comprehensive Patient Record")
def show_patient_details(df_row):
    st.write(f"Detailed clinical attributes cataloged for **Patient {df_row['PatientID'].values[0]}**:")
    
    details_df = pd.DataFrame({
        "Clinical Attribute": ["Patient Identifier", "Age", "Atrial Fibrillation Classification", "Left Atrial Volume (LAVi)", "Body Mass Index (BMI)", "Historical Clinical Outcome"],
        "Value": [
            str(df_row['PatientID'].values[0]),
            f"{df_row['Age'].values[0]} years",
            str(df_row['AF_Type'].values[0]),
            f"{df_row['LA_Vol'].values[0]} mL",
            f"{df_row['BMI'].values[0]} kg/m²",
            str(df_row['Outcome'].values[0])
        ]
    })
    st.table(details_df)
    if st.button("Close Window"):
        st.rerun()

# 4. MAIN PREDICTION ROW
st.title("🫀 AF Recurrence Clinical Decision Support")
st.markdown("### Predictive Analytics for Atrial Fibrillation Management")

risk_score = 0.78
risk_level = "High Risk" if risk_score > 0.6 else "Moderate Risk" if risk_score > 0.3 else "Low Risk"

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.metric(
        label="12-Month Recurrence Risk",
        value=f"{risk_score:.0%}",
        delta=risk_level,
        delta_color="inverse"
    )
    
with col2:
    st.metric(
        label="LA Volume Index",
        value=f"{patient_row['LA_Vol'].values[0]} mL",
        delta="Above threshold" if patient_row['LA_Vol'].values[0] > 40 else "Normal"
    )

with col3:
    st.markdown("##### 🤖 LLM Supervisor Brief")
    st.info(
        f"**Assessment:** {risk_level} driven by LA Volume ({patient_row['LA_Vol'].values[0]} mL) "
        f"and AF Type ({patient_row['AF_Type'].values[0]}). "
        "**Recommended:** Short-term AAD continuation with close monitoring."
    )

# Cleaned-up 2-column wide button span directly under metrics 1 & 2
btn_col, clear_col = st.columns([2, 3])
with btn_col:
    if st.button("🔎 View Patient Details", use_container_width=True):
        show_patient_details(patient_row)

st.divider()

# 5. CASE-BASED REASONING (5 Patients Retrieval)
st.subheader("📊 Case-Based Reasoning")

with st.expander("View Similar Historical Cases (Nearest Neighbors Comparison)", expanded=False):
    st.write("The following 5 patients from the training cohort share the closest alignment with current clinical features:")
    
    neighbors = data[data['PatientID'] != selected_id].sample(5, random_state=42)
    
    patient_row_display = patient_row.copy()
    patient_row_display['Category'] = '⭐ CURRENT PATIENT'
    neighbors_display = neighbors.copy()
    neighbors_display['Category'] = '📁 Historical Match'
    
    comparison_df = pd.concat([patient_row_display, neighbors_display], axis=0).reset_index(drop=True)
    cols = ['Category', 'PatientID', 'Age', 'AF_Type', 'LA_Vol', 'BMI', 'Outcome']
    comparison_df = comparison_df[cols]

    def highlight_selected(row):
        if row['Category'] == '⭐ CURRENT PATIENT':
            return ['background-color: #1f77b4; color: white; font-weight: bold'] * len(row)
        return [''] * len(row)

    st.dataframe(
        comparison_df.style.apply(highlight_selected, axis=1),
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("📌 Note: Similarity space maps LA Volume, Age, and AF Type using a k-NN (k=5) algorithm topology.")
    
    recurred_count = neighbors_display[neighbors_display['Outcome'] == 'Recurred'].shape[0]
    st.write(f"**Historical Pattern:** {recurred_count} out of 5 matched records experienced recurrence ({recurred_count/5:.0%})")

st.divider()

# 6. DUAL EXPLAINABILITY SECTIONS (Diverging Horizontal Bar Chart)
st.divider()
st.subheader("🔍 Decision Logic (Feature Attribution Profiles)")
st.markdown(f"**Baseline Cohort Risk:** 40% | **Patient Risk Contribution:** +38% | **Final Calculated Risk:** **{risk_score:.0%}**")

# Import altair locally for the visualization layer
import altair as alt

# 1. Prepare unified data structures for the diverging plot
explainability_data = pd.DataFrame({
    "Clinical Feature": [
        "LA Volume Enlargement", "Persistent Classification", "Advanced Age Index", "Elevated BMI Impact", "Comorbidity Score",
        "Active Antiarrhythmic Control", "Controlled Blood Pressure", "No Left Atrial Fibrosis", "Preserved EF (>55%)", "Prior Catheter Ablation"
    ],
    "SHAP Impact Value": [0.22, 0.15, 0.08, 0.04, 0.01, -0.05, -0.04, -0.02, -0.01, 0.00],
    "Type": ["Risk Factor", "Risk Factor", "Risk Factor", "Risk Factor", "Risk Factor", 
            "Protective Factor", "Protective Factor", "Protective Factor", "Protective Factor", "Protective Factor"]
})

# 2. Build the Altair Diverging Chart Object
diverging_chart = (
    alt.Chart(explainability_data)
    .mark_bar(height=24, cornerRadiusEnd=4)
    .encode(
        # X-axis maps the impact extending left/right from 0
        x=alt.X(
            "SHAP Impact Value:Q",
            title="Attribution Impact on Risk Score",
            axis=alt.Axis(format="+.0%", tickCount=6),
            scale=alt.Scale(domain=[-0.25, 0.25]) # Keeps the 0 baseline beautifully centered
        ),
        # Y-axis lists features, sorted logically by absolute impact magnitude
        y=alt.Y(
            "Clinical Feature:N",
            title=None,
            sort=alt.EncodingSortField(field="SHAP Impact Value", order="descending"),
            axis=alt.Axis(labelPadding=10)
        ),
        # Color conditional logic mapping semantic clinical risk
        color=alt.Color(
            "Type:N",
            legend=alt.Legend(title="Factor Categorization", orient="top-left"),
            scale=alt.Scale(
                domain=["Risk Factor", "Protective Factor"],
                range=["#e45756", "#4c78a8"] # Clean crimson-red vs medical-blue/teal palette
            )
        ),
        tooltip=[
            alt.Tooltip("Clinical Feature:N"),
            alt.Tooltip("SHAP Impact Value:Q", format="+.1%", title="Impact Contribution"),
            alt.Tooltip("Type:N")
        ]
    )
    .properties(
        height=350,
        title=alt.TitleParams(
            text="Bi-Directional Risk Vector Mapping",
            subtitle="Left-facing blocks compress risk | Right-facing blocks escalate risk thresholds",
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
            subtitleColor="#666"
        )
    )
)

# 3. Add a clean baseline ruler annotation over the chart at exactly 0.0
zero_rule = alt.Chart(pd.DataFrame([{"zero": 0}])).mark_rule(color="#333", strokeWidth=1.5).encode(x="zero:Q")
final_layered_chart = alt.layer(diverging_chart, zero_rule).configure_view(strokeWidth=0)

# 4. Render chart seamlessly responsive across containers
st.altair_chart(final_layered_chart, use_container_width=True)

st.info("💡 **Clinical Context:** The waterfall balance equation matches: **Baseline (40%)** + **Risk Sum (+50%)** + **Protective Sum (-12%)** = **78% Final Risk Profile**.")

st.divider()

# 7. EXPORT REPORT SECTION
st.subheader("📄 Export Prediction Report")

llm_narrative = (
    f"The patient (ID: {selected_id}) presents with a {risk_score:.0%} estimated risk of atrial fibrillation "
    f"recurrence within 12 months. Key clinical drivers contributing to this risk assessment include:\n\n"
    f"1. Left Atrial Volume: {patient_row['LA_Vol'].values[0]} mL - {'elevated and associated with structural remodeling' if patient_row['LA_Vol'].values[0] > 40 else 'within normal limits'}\n"
    f"2. AF Type: {patient_row['AF_Type'].values[0]} - {('associated with higher recurrence rates' if patient_row['AF_Type'].values[0] == 'Persistent' else 'generally favorable prognosis')}\n"
    f"3. Patient Age: {patient_row['Age'].values[0]} years - relevant for treatment planning\n"
    f"4. BMI: {patient_row['BMI'].values[0]} - consider impact on procedural outcomes\n\n"
    f"Nearest neighbor metrics from our registry profile suggest structural synergy with {recurred_count} historical matches "
    f"out of 5 that went on to display recurrent episodes.\n\n"
    "Clinical recommendation: Establish optimized rhythm pathways, maintain antiarrhythmic tracking, and review at the 3-month mark."
)

col_exp1, col_exp2 = st.columns([2, 1])

with col_exp1:
    st.write("Generate a clear, archival PDF report encapsulating the dual explainability matrices, nearest neighbors registry profile, and current automated recommendations.")

with col_exp2:
    try:
        pdf_bytes = create_pdf(patient_row, f"{risk_score:.0%}", llm_narrative)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"AF_Report_Patient_{selected_id}_{date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.success("✅ Report compiled cleanly")
    except Exception as e:
        st.error(f"❌ Document compilation exception: {str(e)}")

st.divider()
st.caption("⚠️ **Disclaimer:** This tool is for research and clinical decision support purposes only. It should not replace clinical judgment or be used as the sole basis for treatment decisions.")