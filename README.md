# 🧮 Insurance Charges Prediction Model

This project builds a **Linear Regression model** to predict medical insurance charges based on demographic and health-related features such as age, BMI, number of children, smoking status, sex, and region. The model uses **feature scaling** (`StandardScaler`) to improve performance and stability.

---

## 🔍 Overview

- **Problem:** Predict insurance charges given personal and lifestyle attributes.
- **Dataset:** Modified version of the Medical Cost Personal Datasets.
- **Target Variable:** `charges` (annual medical insurance cost).
- **Model Used:** Linear Regression (with StandardScaler).
- **Evaluation Metrics:**
  - R² Score
  - Mean Absolute Error (MAE)
- **Tool:** Kaggle Notebook (Python, scikit-learn, pandas, seaborn, matplotlib)

---

## ⚙️ Features Used

- `age`
- `bmi`
- `children`
- `sex_male`
- `smoker_yes`
- `region_northwest`
- `region_southeast`
- `region_southwest`

> Note: One-hot encoding was applied to categorical features.

---

## 📈 Model Workflow

1. **Data Preprocessing**
   - Drop target (`charges`) from features.
   - One-hot encode categorical variables.
   - Standardize features using `StandardScaler`.

2. **Train/Test Split**
   - 80% training, 20% testing.

3. **Model Training**
   - Linear Regression fitted on scaled features.

4. **Evaluation**
   - Predicted charges compared with actual using R² and MAE.
   - Scatter plot to visualize prediction accuracy.

5. **Prediction**
   - Includes prediction on a sample input (e.g., a 70-year-old smoker).

---

## 📊 Sample Prediction

```python
sample = pd.DataFrame([{
    'age': 70,
    'bmi': 28.5,
    'children': 2,
    'sex_male': 1,
    'smoker_yes': 1,
    'region_northwest': 1,
    'region_southeast': 0,
    'region_southwest': 0
}])

sample_scaled = scaler.transform(sample)
predicted_charge = model.predict(sample_scaled)

print(f"Predicted Insurance Charge: ${predicted_charge[0]:.2f}")
