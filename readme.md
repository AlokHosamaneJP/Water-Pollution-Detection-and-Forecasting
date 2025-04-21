# 💧 Water Pollution Detection and Forecasting

This repository contains a complete machine learning pipeline to **predict water potability** based on various physicochemical properties. The project includes **data preprocessing**, **feature engineering**, **model training with hyperparameter tuning**, **explainable AI**, and an interactive **Streamlit dashboard** for real-time predictions.

---

## 📌 Project Summary

Clean water is a fundamental human need. This project leverages data science to determine whether a water sample is **potable (safe to drink)** or **non-potable** based on measurable chemical attributes.

> 🔎 **Model Accuracy**: ~68% on the test set using a tuned `RandomForestClassifier`.

---

## 📂 Data Source

The dataset used is `water_potability 2.csv` and contains the following features:

| Feature            | Description                                     |
|--------------------|-------------------------------------------------|
| `ph`               | Acidity/alkalinity level of the water           |
| `Hardness`         | Level of calcium and magnesium salts            |
| `Solids`           | Total dissolved solids (ppm)                    |
| `Chloramines`      | Disinfectant used in water treatment            |
| `Sulfate`          | Concentration of sulfate ions                   |
| `Conductivity`     | Water’s ability to conduct electricity          |
| `Organic_carbon`   | Organic matter content                          |
| `Trihalomethanes`  | Chemical byproduct of disinfection              |
| `Turbidity`        | Water clarity (suspended particles)             |
| `Potability`       | Target (1 = potable, 0 = not potable)           |

---

## 🧠 ML Pipeline

### 1. **Preprocessing**
- Handled missing values using mean imputation.
- Normalized features for consistent scaling.
- Removed outliers using IQR-based filtering.

### 2. **Balancing**
- Used `RandomUnderSampler` from `imbalanced-learn` to balance the dataset and improve prediction fairness between potable and non-potable classes.

### 3. **Modeling**
- Algorithm: `RandomForestClassifier`
- Optimized via `GridSearchCV` with 5-fold cross-validation.
- Achieved best results with 70% accuracy


### 4. **Explainable AI**
🔍 Explainable AI: SHAP & LIME
To ensure transparency and trust in the model's predictions, this project incorporates Explainable AI (XAI) techniques using SHAP (SHapley Additive Explanations) and LIME (Local Interpretable Model-agnostic Explanations). These tools help interpret why the model predicted a water sample as potable or non-potable.

✅ Why Use Explainable AI?
While traditional models can be accurate, they often act as "black boxes," offering little insight into their decision-making processes. SHAP and LIME address this by providing human-interpretable visual explanations, enabling:

Better understanding of model behavior

Increased stakeholder trust and accountability

Debugging and validation of data/model inconsistencies

📊 SHAP (SHapley Additive Explanations)
Global Insight: Identifies the most influential features across the entire dataset.

Local Explanation: Breaks down how each feature impacts an individual prediction.

Use Case in this Project:
In the context of water potability, SHAP showed how features like ph, Trihalomethanes, or Sulfate pushed predictions toward either safe or unsafe.

📌 Example: A high sulfate level might strongly reduce the model's confidence in a sample being potable — SHAP highlights this in the waterfall plot.

📈 LIME (Local Interpretable Model-Agnostic Explanations)
Local Explanation: Focuses on interpreting one prediction at a time by approximating the model locally with a simpler, interpretable model (like linear regression).

On-Demand Interpretability: Especially useful during live user interactions in the Streamlit dashboard.

Use Case in this Project:
When a user enters water quality parameters, LIME explains the top 5 reasons behind the model's "potable" or "non-potable" decision in real-time.

📌 Example: If a user inputs a high Chloramines value, LIME might show that it negatively influenced the decision, helping users understand what needs to be improved in the water sample.

🧠 Summary

Tool	Scope	Purpose
SHAP	Global + Local	Quantifies and visualizes feature contributions
LIME	Local	Simplifies and explains individual predictions


### 5. **Deployment**
- Streamlit-based frontend (`streamlit_app.py`) for real-time, on-demand classification.

---

## 📈 Dashboard Preview

The dashboard allows users to:
- Enter chemical metrics of a water sample.
- Click a button to predict its potability.
- Get instant feedback on the safety of the input sample.

> ✅ Model file: `models/final_water_quality_model.pkl`

---

## 🖥️ How to Run the Project

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/AlokHosamaneJP/Water-Pollution-Detection-and-Forecasting.git
cd Water-Pollution-Detection-and-Forecasting
