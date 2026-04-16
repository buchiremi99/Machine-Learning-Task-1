import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Stroke Risk Prediction",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for compact styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    .main-header {
        background-color: #4a5568;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        font-size: 1.5rem;
        margin: 0;
    }
    .main-header p {
        font-size: 0.8rem;
        margin: 0.25rem 0 0 0;
        opacity: 0.9;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        text-align: center;
    }
    .prediction-high-risk {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
    }
    .prediction-low-risk {
        background-color: #dcfce7;
        border: 2px solid #22c55e;
    }
    .prediction-box h2 {
        font-size: 1.25rem;
        margin: 0;
    }
    .prediction-box p {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    .metric-card {
        background-color: white;
        padding: 0.5rem;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-card h3 {
        font-size: 1.1rem;
        margin: 0;
    }
    .metric-card p {
        font-size: 0.7rem;
        margin: 0;
        color: #6b7280;
    }
    .stButton > button {
        background-color: #4a5568;
        color: white;
        font-weight: bold;
        padding: 0.4rem 1rem;
        border-radius: 6px;
        border: none;
        font-size: 0.9rem;
    }
    .stButton > button:hover {
        background-color: #2d3748;
    }
    .form-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .center-button {
            background-color: green;
        display: flex;
        justify-content: center;
        margin-top: 0.75rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.3rem;
    }
    .stSelectbox label, .stNumberInput label {
        font-size: 0.8rem;
        font-weight: normal;
    }
    div[data-testid="stSelectbox"], div[data-testid="stNumberInput"] {
        margin-bottom: 0.3rem;
    }
    hr {
        margin: 0.5rem 0;
    }
    .risk-meter-label {
        font-size: 0.8rem;
        font-weight: bold;
        text-align: center;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('lr_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Title
st.markdown("""
<div class="main-header">
    <h1>Stroke Risk Prediction</h1>
    <p>Machine Learning Model for Early Stroke Detection</p>
</div>
""", unsafe_allow_html=True)

# Form container
with st.container():

    
    # Two-column layout for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox(
            "Gender",
            options=["Female", "Male", "Other"]
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=0.0,
            max_value=120.0,
            value=45.0,
            step=1.0
        )
        
        hypertension = st.selectbox(
            "Hypertension",
            options=["No", "Yes"]
        )
        
        heart_disease = st.selectbox(
            "Heart Disease",
            options=["No", "Yes"]
        )
        
        avg_glucose_level = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=50.0,
            max_value=300.0,
            value=100.0,
            step=1.0
        )
        
        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )
    
    with col2:
        ever_married = st.selectbox(
            "Marital Status",
            options=["No", "Yes"]
        )
        
        work_type = st.selectbox(
            "Work Type",
            options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
        )
        
        residence_type = st.selectbox(
            "Residence Type",
            options=["Urban", "Rural"]
        )
        
        smoking_status = st.selectbox(
            "Smoking Status",
            options=["never smoked", "formerly smoked", "smokes", "Unknown"]
        )
    
    # Predict button centered under the form
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    predict_button = st.button("Predict Stroke Risk", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    


# Convert inputs to model format
def prepare_input_data():
    """Convert UI inputs to DataFrame format expected by model"""
    data = {
        'gender': gender.lower(),
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'ever_married': ever_married,
        'work_type': work_type.lower(),
        'Residence_type': residence_type.lower(),
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi if bmi > 0 else np.nan,
        'smoking_status': smoking_status
    }
    return pd.DataFrame([data])

# Make prediction when button is clicked
if predict_button:
    try:
        # Load model
        model = load_model()
        
        # Prepare input data
        input_df = prepare_input_data()
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Display results
        st.markdown("### Prediction Results")
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Age", f"{age:.0f}")
        with col2:
            st.metric("BMI", f"{bmi:.1f}" if bmi > 0 else "N/A")
        with col3:
            st.metric("Glucose", f"{avg_glucose_level:.0f}")
        with col4:
            st.metric("Risk", f"{probability*100:.1f}%")
        
        # Prediction result box
        if prediction == 1:
            st.markdown("""
            <div class="prediction-box prediction-high-risk">
                <h2>HIGH RISK OF STROKE</h2>
                <p>The model predicts an elevated risk of stroke.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box prediction-low-risk">
                <h2>LOW RISK OF STROKE</h2>
                <p>The model predicts low risk of stroke.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk meter
        risk_level = probability
        if risk_level < 0.3:
            color = "#22c55e"
            level = "Low Risk"
        elif risk_level < 0.6:
            color = "#eab308"
            level = "Moderate Risk"
        else:
            color = "#ef4444"
            level = "High Risk"
        
        st.markdown(f"""
        <div style="background-color: #e5e7eb; border-radius: 6px; height: 12px; margin: 8px 0;">
            <div style="background-color: {color}; width: {risk_level*100}%; height: 12px; border-radius: 6px;"></div>
        </div>
        <div class="risk-meter-label">{level} ({probability*100:.1f}%)</div>
        """, unsafe_allow_html=True)
        
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

else:
    # Model performance metrics
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>84.5%</h3>
            <p>ROC-AUC</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>80.0%</h3>
            <p>Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown("""
        <div class="metric-card">
            <h3>74.8%</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col4:
        st.markdown("""
        <div class="metric-card">
            <h3>23.7%</h3>
            <p>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
