import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import date

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
    
    # Patient Demographics Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Patient ID: {patient_data['PatientID']}", ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 7, f"Age: {patient_data['Age']} | Type: {patient_data['AF_Type']} | BMI: {patient_data['BMI']}", ln=1)
    pdf.ln(5)
    
    # Risk Assessment Section
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Calculated Risk of 12-Month Recurrence: {risk_score}", ln=1, fill=True)
    pdf.ln(5)
    
    # Clinical Narrative Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Clinical Narrative & Supervisor Analysis:", ln=1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, narrative)
    
    return pdf.output()

# 1. SETUP
st.set_page_config(page_title="AF Recurrence Supervisor", layout="wide")

@st.cache_data
def load_and_prep_data():
    # Simulated Registry
    df = pd.DataFrame({
        'PatientID': range(101, 120),
        'Age': np.random.randint(45, 80, 19),
        'AF_Type': np.random.choice(['Paroxysmal', 'Persistent'], 19),
        'LA_Vol': np.random.randint(25, 60, 19),
        'BMI': np.random.randint(22, 38, 19),
        'Outcome': np.random.choice(['Recurred', 'Sinus Rhythm'], 19)
    })
    return df

data = load_and_prep_data()

# 2. SIDEBAR SELECTION
st.sidebar.header("Patient Selection")
selected_id = st.sidebar.selectbox("Select Patient ID", data['PatientID'])
patient_row = data[data['PatientID'] == selected_id].copy()

# 3. MAIN PREDICTION ROW
st.title("AF Recurrence Clinical Decision Support")

risk_score = 78%

col1, col2 = st.columns([1, 2])
with col1:
    st.metric(label="12-Month Recurrence Risk", value="78%", delta="High Risk", delta_color="inverse")
    
with col2:
    st.subheader("LLM Supervisor Brief")
    st.info("Assessment: High risk driven by LA Volume and AF Type. Recommended: Short-term AAD continuation.")

st.divider()

# 4. SIMILAR HISTORICAL CASES (Expander Row)
# Logic to create a comparison dataframe
st.subheader("Case-Based Reasoning")

with st.expander("View Similar Historical Cases (Nearest Neighbors Comparison)", expanded=False):
    st.write("The following patients from the training set share the most similar clinical features:")
    
    # Simulate finding 3 neighbors (In your real code, use sklearn.neighbors here)
    neighbors = data[data['PatientID'] != selected_id].sample(3)
    
    # Combine Selected Patient with Neighbors for direct comparison
    # We add a 'Category' column to distinguish them
    patient_row['Category'] = 'CURRENT PATIENT'
    neighbors['Category'] = 'Historical Match'
    
    comparison_df = pd.concat([patient_row, neighbors], axis=0).reset_index(drop=True)
    
    # Reorder columns to put Category first
    cols = ['Category'] + [c for c in comparison_df.columns if c != 'Category']
    comparison_df = comparison_df[cols]

    # Styling: Highlight the 'CURRENT PATIENT' row
    def highlight_selected(s):
        return ['background-color: #1f77b4; color: white' if s.Category == 'CURRENT PATIENT' else '' for _ in s]

    st.dataframe(
        comparison_df.style.apply(highlight_selected, axis=1),
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("Note: Similarity is calculated based on LA Volume, Age, and AF Type using a k-NN (k=3) algorithm.")

# 5. SHAP / EXPLAINABILITY ROW
st.divider()
st.subheader("Decision Logic (SHAP Explanations)")
st.write("Internal Logic Visualization would go here.")
#

st.divider()
st.subheader("Export Prediction Report")

# Mock narrative for the PDF
llm_narrative = (
    f"The patient (ID: {selected_id}) presents with a {risk_score:.1%} risk of recurrence. "
    "Key drivers include LA Volume and existing comorbidities. Nearest neighbor analysis "
    "suggests a high correlation with historical cases that failed within 6 months. "
    "Recommendation: Consider aggressive rhythm control and early 3-month follow-up."
)

# Create PDF and provide download button
pdf_bytes = create_pdf(patient_row, f"{risk_score:.1%}", llm_narrative)

st.download_button(
    label="📥 Download Clinical Risk Report (PDF)",
    data=pdf_bytes,
    file_name=f"AF_Report_Patient_{selected_id}.pdf",
    mime="application/pdf"
)