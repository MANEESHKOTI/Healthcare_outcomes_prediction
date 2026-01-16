# 🧠 NeuroGuard AI: Stroke Risk Prediction & Clinical Decision System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Streamlit-1.31-red)
![ML](https://img.shields.io/badge/XGBoost-2.0-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

🔴 **Live Demo:** [Click here to view the App](🔴 **Live Demo:** [Click here to view the App](https://ecommerce-churn-predictiongit-qr4xwis3jph4rmfas48jdu.streamlit.app/))

---

## 📋 Executive Overview

**NeuroGuard AI** is an end-to-end machine learning system designed to predict **stroke probability** in patients.  
Unlike standard academic ML projects, this system integrates:

- Clinical safety override logic
- Financial impact and ROI analysis
- Explainable AI–ready design

The system shifts healthcare from **reactive** to **preventative**, potentially saving **$35,000+ per high-risk patient** in avoided emergency and long-term care costs.

---

## 🏗️ Project Architecture

Healthcare_outcomes_prediction/
│
├── data/
│ ├── raw/
│ │ └── healthcare-dataset-stroke-data.csv
│ └── processed/
│ └── MICE-imputed and feature-engineered datasets
│
├── notebooks/
│ ├── 01_eda_and_stats.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modeling_and_evaluation.ipynb
│ └── 04_cost_effectiveness_analysis.ipynb
│
├── dashboard/
│ ├── app.py
│ ├── model.pkl
│ └── scaler.pkl
│
├── reports/
│ └── Executive_Summary.md
│
└── requirements.txt

---

## 🔬 Methodology & Technical Approach

### 1. Data Cleaning & Imputation

- **Missing Values**  
  BMI values were imputed using **MICE (Multiple Imputation by Chained Equations)** based on age, gender, and glucose levels.

- **Outlier Handling**  
  Clinical outliers such as glucose > 250 mg/dL and BMI > 50 were retained, as they represent high-risk pathological cases.

---

### 2. Feature Engineering

Domain-informed features were created to model non-linear medical risk:

- **Risk_Score**  
  Composite indicator of hypertension, heart disease, obesity, and diabetes.

- **Age_Glucose_Interaction**  
  Captures compounding vascular risk in elderly patients.

- **BMI_Category & Glucose_Risk**  
  WHO-standardized categorical risk buckets.

---

### 3. Modeling Strategy

Two models were evaluated:

| Model | Observation |
|------|------------|
| MLP Neural Network | Baseline performance, sensitive to imbalance |
| XGBoost | High recall, stable, clinically reliable |

- **Class Imbalance** handled using **SMOTE**
- **Final Performance**: AUC-ROC > 0.85 with recall prioritized

---

## 🛡️ Clinical Safety Layer

To prevent false negatives in critical patients, a rule-based override was implemented:

IF model_probability < 0.15
AND (heart_disease = TRUE OR age > 80 OR glucose > 230)
THEN risk_level = "MODERATE"

This ensures no high-risk patient is mistakenly labeled as low risk due to statistical noise.

---

## 📊 Business Impact

| Metric | Reactive Care | NeuroGuard AI | Impact |
|------|---------------|---------------|--------|
| Cost per Event | $40,000 | $2,000 | 95% Reduction |
| Detection | Symptom-based | Screening-based | Earlier Intervention |
| ROI | N/A | > 1500% | Strong Positive |

Simulation results indicate savings of **$1.5M–$2M** per 100 high-risk patients identified early.

---

## 🚀 Installation & Usage

### Local Setup

git clone https://github.com/MANEESHKOTI/Healthcare_outcomes_prediction.git
cd Healthcare_outcomes_prediction
pip install -r requirements.txt
streamlit run dashboard/app.py

---

### Reproducing the Analysis

Run notebooks in sequence:

01_eda_and_stats
02_feature_engineering
03_modeling_and_evaluation
04_cost_effectiveness_analysis

---

## ⚠️ Disclaimer

NeuroGuard AI is a **clinical decision-support system**, not a diagnostic replacement.  
Final decisions must always be made by qualified medical professionals.
