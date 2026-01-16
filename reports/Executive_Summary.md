# Executive Summary: Stroke Prediction & Cost Containment Strategy

## 1. Project Overview
This project aimed to develop a machine learning solution to predict stroke risk in patients using the `healthcare-dataset-stroke-data`. The primary business goal was to reduce healthcare costs by identifying high-risk patients early, enabling preventative intervention ($2,000) rather than reactive emergency care ($40,000).

## 2. Key Findings (Data & Analysis)
* **Risk Factors:** Exploratory Data Analysis (EDA) confirmed that **Age**, **Average Glucose Level**, and **BMI** are the strongest predictors of stroke.
* **Hypothesis Testing:**
    * *Glucose:* Statistically significant difference found between stroke/non-stroke patients (p < 0.05).
    * *Age:* Stroke risk doubles significantly in the 'Senior' age category (>65).
* **Data Quality:** Addressed class imbalance (Stroke < 5%) using SMOTE and imputed missing BMI values using MICE (Multiple Imputation).

## 3. Model Performance
Two advanced algorithms were compared: **XGBoost (Ensemble)** vs. **MLP Classifier (Neural Network)**.

* **Winner:** XGBoost
* **Performance:** Achieved an AUC-ROC of >0.85 (Training) and robust recall on the test set.
* **Why XGBoost?** It outperformed the Neural Network in handling tabular data and offered superior interpretability via SHAP values.

## 4. Cost-Benefit Analysis (Business Impact)
Using a synthetic cost model:
* **Cost of Event:** $40,000 (Stroke treatment)
* **Cost of Prevention:** $2,000 (Medication/Monitoring)

**Result:** The model-based intervention strategy is projected to save approximately **$20,000 - $35,000 per 100 high-risk patients** compared to a reactive baseline, provided the intervention success rate is >50%.

## 5. Recommendations
1.  **Deploy Dashboard:** Utilize the `NeuroGuard AI` dashboard in triage centers to flag high-risk incoming patients.
2.  **Targeted Screening:** Prioritize patients over 60 with Glucose > 200mg/dL for immediate preventative scans, as this specific interaction showed the highest SHAP impact.