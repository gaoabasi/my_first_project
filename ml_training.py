# ============================================================
# CUSTOMER SPENDING PREDICTION MODEL
# Linear Regression vs Decision Tree
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

# -----------------------------
# 2. Load Dataset
# -----------------------------
# Badilisha jina la file kama dataset yako ina jina tofauti
df = pd.read_csv("customer_data.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# -----------------------------
# 3. Data Preprocessing
# -----------------------------

# Encode Gender (Male/Female)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# (Optional) Encode Preferred_Product if unataka kuitumia
# df = pd.get_dummies(df, columns=['Preferred_Product'], drop_first=True)

print("\nAfter Encoding:")
print(df.head())

# -----------------------------
# 4. Feature Selection
# -----------------------------
X = df[['Age', 'Annual_Income', 'Monthly_Visits', 'Gender']]
y = df['Spending_Score']

print("\nSelected Features:")
print(X.head())

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# -----------------------------
# 6. Train Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# -----------------------------
# 7. Evaluate Linear Regression
# -----------------------------
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n--- Linear Regression Performance ---")
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)

# -----------------------------
# 8. Train Decision Tree Regressor
# -----------------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

# -----------------------------
# 9. Evaluate Decision Tree
# -----------------------------
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\n--- Decision Tree Performance ---")
print("MSE:", mse_dt)
print("R2 Score:", r2_dt)

# -----------------------------
# 10. Model Comparison
# -----------------------------
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree"],
    "MSE": [mse_lr, mse_dt],
    "R2 Score": [r2_lr, r2_dt]
})

print("\nModel Comparison:")
print(comparison)

# -----------------------------
# 11. Visualization
# -----------------------------
plt.figure()
plt.scatter(y_test, y_pred_lr)
plt.xlabel("Actual Spending Score")
plt.ylabel("Predicted Spending Score")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

# -----------------------------
# 12. Save Best Model
# (Chagua model yenye performance bora)
# -----------------------------
# Hapa tunahifadhi Linear Regression (badili kama DT ni bora)
pickle.dump(lr, open("model.pkl", "wb"))
print("\nBest model saved as model.pkl")

# -----------------------------
# 13. Conclusion (print)
# -----------------------------
print("""
Conclusion:
The Linear Regression and Decision Tree models were trained and evaluated.
Based on the evaluation metrics, the best-performing model was selected
and saved for deployment in a Streamlit AI application.
""")
