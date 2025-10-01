# Doordash Delivery Time Prediction 🚴‍♂️⏱️

This project builds and evaluates machine learning models to predict the **delivery time** of Doordash orders, from the moment the order is placed until it reaches the customer. The work focuses on understanding **operational bottlenecks** (driver supply, restaurant prep time, distance, and order complexity) and improving ETA accuracy for a better customer and dasher experience.

---

## 📂 Project Overview
- **Goal:** Predict the total delivery time of an order with minimal bias and improved trust (reduce underestimation of ETA).
- **Approach:**  
  1. **Data Cleaning & Feature Engineering**
     - Price-based features: price range, volatility, menu complexity
     - Supply-demand metrics: orders-to-available-dashers, non-taken orders
     - Delivery dynamics: estimated driving time, delivery difficulty, delivery speed
     - Store-level historical features: average prep & delivery times
     - Temporal features: hour of day, peak-hour flags
  2. **Modeling**
     - Baseline: XGBoost regression with MAE objective
     - Advanced: Quantile Regression (70th percentile) to reduce underestimation bias
     - Feature importance analysis (SHAP, permutation importance)
  3. **Evaluation**
     - Metrics: MAE, underestimation rate, residual analysis
     - Error slicing: by distance, peak hours, store history, and dasher availability
     - Visualizations: residual plots, feature-target relationships, prediction vs actual histograms

---

## 🚀 Key Results
- **Baseline MAE:** ~9.6 minutes  
- **After feature engineering:** ~8.6 minutes

Top features influencing delivery times:
- If it's peak hour.
- log of the the estimated store to consumer duration. 
- The delivery difficulty if it is a rush time.
- The total orders that are not taken.
- The amount of available dashers.
- The historical average store place duration.

---

## 🛠️ Tech Stack
- **Python**: pandas, numpy, scikit-learn
- **Modeling**: XGBoost, LightGBM
- **Visualization**: matplotlib, seaborn
- **Interpretability**: feature importance

---

## 📊 Example Visualization
- Residuals vs Features (identify bias regions)
- Predicted vs Actual distribution overlap
- Feature importances (SHAP summary)
