import streamlit as st
import pandas as pd
import numpy as np

# 1. SETUP & MOCK DATA
st.set_page_config(page_title="AF Recurrence Predictor", layout="wide")

@st.cache_data
def load_mock_data():
    # In reality, this would be your cleaned registry CSV
    df = pd.DataFrame({
        'PatientID': range(101, 111),
        'Age': np.random.randint(45, 80, 10),
        'AF_Type': np.random.choice(['Paroxysmal', 'Persistent'], 10),
        'LA_Vol': np.random.randint(25, 60, 10),
        'BMI': np.random.randint(22, 38, 10)
    })
    return df

data = load_mock_data()

# 2. NEAREST NEIGHBOR POPUP (MODAL)
@st.dialog("Similar Historical Cases")
def show_neighbor_details(patient_id):
    st.write(f"Retrieving cases most similar to **Patient {patient_id}**...")
    
    # Placeholder for your k-NN logic
    # neighbors = knn_model.kneighbors(current_patient_features)
    mock_neighbors = data.sample(3) 
    
    st.table(mock_neighbors)
    st.info("These patients shared similar LA volumes and AF types. 2/3 recurred within 12 months.")

# 3. SIDEBAR: PATIENT SELECTION
st.sidebar.header("Clinical Input")
selected_id = st.sidebar.selectbox("Select Patient ID", data['PatientID'])
patient_row = data[data['PatientID'] == selected_id].iloc[0]

# 4. MAIN DASHBOARD
st.title("Atrial Fibrillation Post-Ablation Risk Supervisor")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Profile")
    st.write(f"**Age:** {patient_row['Age']}")
    st.write(f"**AF Type:** {patient_row['AF_Type']}")
    st.write(f"**LA Volume Index:** {patient_row['LA_Vol']} ml/m²")
    
    # Trigger the Popup
    if st.button("🔍 View Similar Patients"):
        show_neighbor_details(selected_id)

with col2:
    st.subheader("AI Risk Assessment")
    risk_score = np.random.random()  # Replace with your model.predict_proba()
    
    # Visual Feedback
    st.metric(label="12-Month Recurrence Risk", value=f"{risk_score:.1%}")
    st.progress(risk_score)

st.divider()

# 5. LLM SUPERVISOR NARRATIVE
st.subheader("Clinical Narrative (LLM Supervisor)")
with st.expander("View Full Report", expanded=True):
    # This is where your LangChain / LLM call would go
    st.markdown(f"""
    **Assessment:** This patient presents with {patient_row['AF_Type']} AF and a risk score of {risk_score:.1%}.
    
    **Reasoning:** The primary driver is the LA Volume of {patient_row['LA_Vol']}. 
    Historically, patients with this profile benefit from intensive rhythm control follow-up.
    """)