import streamlit as st
import pandas as pd
import numpy as np

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