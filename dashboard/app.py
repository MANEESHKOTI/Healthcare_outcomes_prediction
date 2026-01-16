import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import warnings

# --- 0. Setup & Config ---
warnings.filterwarnings("ignore") # Silence all warnings for clean UI
st.set_page_config(page_title="NeuroGuard AI | Clinical Dashboard", layout="wide", page_icon="🧠")

# Custom CSS for "Goated" Interface
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white; font-weight: bold; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 5px solid #FF4B4B;
    }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- 1. Load Resources ---
@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    data_path = os.path.join(current_dir, '../data/processed/dashboard_data.csv')
    
    try:
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        df = pd.read_csv(data_path)
        return model, scaler, df
    except FileNotFoundError:
        st.error("⚠️ System Error: Model/Data files missing. Run notebooks first.")
        st.stop()

model, scaler, df_ref = load_resources()

# --- 2. Header ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🧠 NeuroGuard AI")
    st.markdown("**Clinical Decision Support System** | Stroke Risk Prediction & Cost Analysis")
with c2:
    st.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=70)

st.markdown("---")

# --- 3. Medical Input Form ---
with st.form("medical_form"):
    st.subheader("📝 Patient Vitals")
    
    # Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1: age = st.slider("Age", 0, 100, 60)
    with c2: glucose = st.number_input("Avg Glucose", 50.0, 300.0, 150.0)
    with c3: bmi = st.number_input("BMI", 10.0, 60.0, 32.0)
    with c4: gender = st.selectbox("Gender", ["Male", "Female"])

    # Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5: hyper = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with c6: heart = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with c7: work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    with c8: smoke = st.selectbox("Smoking", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    
    # Hidden defaults
    married = "Yes" 
    residence = "Urban"

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 ANALYZE RISK & COSTS")

# --- 4. Logic Core ---
if submitted:
    # 4.1 Preprocessing (Robust Schema)
    raw = pd.DataFrame([{
        'age': age, 'avg_glucose_level': glucose, 'bmi': bmi,
        'hypertension': hyper, 'heart_disease': heart,
        'gender': gender, 'ever_married': married,
        'work_type': work, 'Residence_type': residence, 'smoking_status': smoke
    }])

    # BMI/Glucose/Age Cats
    bmi_cat = 'Obese' if bmi >= 30 else 'Overweight' if bmi >= 25 else 'Normal'
    glu_cat = 'Diabetic' if glucose >= 200 else 'Prediabetic' if glucose >= 140 else 'Normal'
    age_cat = 'Senior' if age >= 65 else 'Middle_Aged' if age >= 45 else 'Young_Adult'
    
    # Risk Score
    risk_score = hyper + heart + (1 if bmi_cat=='Obese' else 0) + (1 if glu_cat=='Diabetic' else 0) + (1 if age_cat=='Senior' else 0)
    interaction = age * glucose

    # Expected Schema
    cols = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 
        'risk_score', 'age_glucose_interaction',
        'gender_Male', 'gender_Other', 'ever_married_Yes', 
        'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
        'Residence_type_Urban', 
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',
        'bmi_cat_Obese', 'bmi_cat_Overweight', 'bmi_cat_Underweight',
        'glucose_cat_Normal', 'glucose_cat_Prediabetic',
        'age_cat_Senior', 'age_cat_Young_Adult'
    ]
    
    final = pd.DataFrame(0, index=[0], columns=cols)
    final['age'] = age; final['hypertension'] = hyper; final['heart_disease'] = heart
    final['avg_glucose_level'] = glucose; final['bmi'] = bmi; final['risk_score'] = risk_score
    final['age_glucose_interaction'] = interaction
    
    def set_bit(val, prefix):
        col = f"{prefix}_{val}"
        if col in final.columns: final[col] = 1
        
    set_bit(gender, "gender"); set_bit(work, "work_type"); set_bit(smoke, "smoking_status")
    set_bit(bmi_cat, "bmi_cat"); set_bit(glu_cat, "glucose_cat"); set_bit(age_cat, "age_cat")

    # 4.2 Prediction (THE GOLDEN OUTPUT)
    input_scaled = scaler.transform(final.values)
    prob = model.predict_proba(input_scaled)[:, 1][0]
    
    # --- 4.3 THE COMMON SENSE "SAFETY LAYER" ---
    # We define "Clinical Overrides" that force a warning even if the model score is low.
    
    override_reason = []
    
    # Rule 1: Heart Disease is never "Low Risk"
    if heart == 1:
        override_reason.append("Patient has history of Heart Disease.")
        
    # Rule 2: Super Senior (>80) is never "Low Risk"
    if age > 80:
        override_reason.append("Patient age (>80) significantly elevates baseline risk.")
        
    # Rule 3: Uncontrolled Diabetes (>230)
    if glucose > 230:
        override_reason.append("Glucose levels indicate uncontrolled diabetes (>230 mg/dL).")

    # Determine Final Display Status
    # If prob is Low (<15%) but Override exists -> Force Upgrade to MODERATE (Yellow)
    
    if prob > 0.4:
        risk_tier = "HIGH"
        color = "#FF4B4B" # Red
        final_msg = "🚨 HIGH RISK DETECTED"
    elif prob > 0.15:
        risk_tier = "MODERATE"
        color = "#FFD700" # Yellow
        final_msg = "⚠️ MODERATE RISK DETECTED"
    else:
        # Model says Low, but do we have overrides?
        if override_reason:
            risk_tier = "MODERATE (Clinical Override)"
            color = "#FFD700" # Yellow
            final_msg = "⚠️ MODERATE RISK (Clinical Override)"
        else:
            risk_tier = "LOW"
            color = "#90EE90" # Green
            final_msg = "✅ LOW RISK DETECTED"

    # --- 5. Dashboard Visuals ---
    st.markdown("---")
    
    row1_1, row1_2 = st.columns([1, 1])
    
    with row1_1:
        st.subheader("1. Clinical Diagnosis")
        
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob * 100,
            title = {'text': "Stroke Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [{'range': [0, 15], 'color': "#e8f5e9"}, {'range': [15, 40], 'color': "#fff9c4"}, {'range': [40, 100], 'color': "#ffebee"}]
            }))
        fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=0,b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Display the Status Message
        if risk_tier == "HIGH":
            st.error(final_msg)
        elif "MODERATE" in risk_tier:
            st.warning(final_msg)
            # SHOW THE OVERRIDE REASON
            if override_reason:
                st.markdown("**Reason for Override:**")
                for reason in override_reason:
                    st.markdown(f"- ❗ {reason}")
        else:
            st.success(final_msg)

    with row1_2:
        st.subheader("2. Business & Financial Impact")
        
        # Cost Logic tied to the Safety Layer (Not just the model)
        if "HIGH" in risk_tier or "MODERATE" in risk_tier:
            cost = 2000
            status = "Preventative Care Required"
            delta_color = "normal" # Grey/Black
            rec = "Administer preventative care & monitoring ($2,000)"
        else:
            cost = 150
            status = "Standard Monitoring"
            delta_color = "off"
            rec = "Routine annual monitoring ($150)"
            
        st.metric(label="Recommended Intervention Cost", value=f"${cost:,}", delta=status, delta_color=delta_color)
        
        st.info(f"**Recommendation:** {rec}")
        st.caption(f"Potential Savings vs Stroke Event ($40k): ${40000 - cost:,}")

    st.markdown("---")
    
    row2_1, row2_2 = st.columns([1, 1])
    
    with row2_1:
        st.subheader("3. Patient vs. Population (Health Radar)")
        
        categories = ['Age', 'Glucose Level', 'BMI', 'Risk Score']
        patient_vals = [age/100, glucose/300, bmi/60, risk_score/5]
        
        stroke_df = df_ref[df_ref['stroke'] == 1]
        avg_vals = [
            stroke_df['age'].mean()/100, 
            stroke_df['avg_glucose_level'].mean()/300, 
            stroke_df['bmi'].mean()/60,
            2.5/5 
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=patient_vals, theta=categories, fill='toself', name='Patient'))
        fig_radar.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='Avg Stroke Case'))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=350, margin=dict(l=40,r=40,t=20,b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with row2_2:
        st.subheader("4. AI Explainability (SHAP)")
        st.markdown("Key factors driving the **Raw Model Score**:")
        
        with st.spinner("Calculating attribution..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)
            
            fig_shap, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0], 
                    base_values=explainer.expected_value, 
                    data=final.iloc[0],
                    feature_names=final.columns
                ),
                max_display=7,
                show=False
            )
            st.pyplot(fig_shap)