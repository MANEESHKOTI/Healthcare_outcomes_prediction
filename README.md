# Healthcare Outcomes Prediction: Stroke Risk AI

## Project Description
A comprehensive end-to-end machine learning project to predict stroke likelihood. This solution includes robust data cleaning (MICE), advanced modeling (XGBoost), cost-benefit analysis, and an interactive Streamlit dashboard for clinical use.

## 📂 Project Structure
```text
healthcare-outcomes-prediction/
│
├── data/
│   ├── raw/                   # Original CSV file
│   └── processed/             # Cleaned & Engineered datasets
├── notebooks/
│   ├── 01_eda_and_stats.ipynb             # Data Cleaning & Hypothesis Testing
│   ├── 02_feature_engineering.ipynb       # Feature Creation (Risk Scores)
│   ├── 03_modeling_and_evaluation.ipynb   # XGBoost vs Neural Net Training
│   └── 04_cost_effectiveness_analysis.ipynb # ROI & Financial Analysis
├── dashboard/
│   ├── app.py                 # Interactive Clinical Dashboard (Streamlit)
│   ├── model.pkl              # Trained Model (Saved Artifact)
│   └── scaler.pkl             # Data Scaler (Saved Artifact)
├── reports/
│   └── Executive_Summary.md   # Final Business Report
└── requirements.txt           # Project Dependencies

##🚀 How to Run
1. Installation

Ensure you have Python 3.9+ installed.

Bash
pip install -r requirements.txt
2. Reproduce the Analysis (Notebooks)

Run the notebooks in order (01 -> 04) to clean data, train models, and generate the cost analysis.

Note: Notebook 03 will automatically save the trained model to the dashboard/ folder.

3. Launch the Dashboard

To use the interactive AI tool:

Bash
streamlit run dashboard/app.py
📊 Key Results
Best Model: XGBoost

Key Insight: Age-Glucose interaction is the critical driver of risk.

Financial Impact: Shift from reactive to preventative care significantly lowers projected hospital costs.