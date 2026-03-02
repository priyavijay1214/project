import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# 1. Generate Synthetic Financial User Data

np.random.seed(42)
n = 500  # number of simulated users

# Feature Engineering
income = np.random.normal(5000, 1000, n)
fixed_ratio = np.random.uniform(0.3, 0.7, n)
month = np.random.randint(1, 13, n)
avg_last_3 = np.random.normal(800, 200, n)
growth_rate = np.random.normal(0.05, 0.02, n)

# Simulated "true" spending behavior
next_month_spending = (
    0.12 * income +
    0.55 * avg_last_3 +
    300 * fixed_ratio +
    45 * growth_rate +
    20 * month +
    np.random.normal(0, 150, n)
)

df = pd.DataFrame({
    "income": income,
    "fixed_ratio": fixed_ratio,
    "month": month,
    "avg_last_3": avg_last_3,
    "growth_rate": growth_rate,
    "next_month_spending": next_month_spending
})

print("Sample Data:")
print(df.head())

# 2. Train/Test Split

X = df.drop("next_month_spending", axis=1)
y = df["next_month_spending"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate Model Performance

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print("MSE:", round(mse, 2))
print("MAE:", round(mae, 2))
print("R^2:", round(r2, 3))

# 5. Interpret Feature Importance

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients:")
print(coefficients)

# 6. Example User Prediction + Recommendation

sample_user = pd.DataFrame({
    "income": [5200],
    "fixed_ratio": [0.5],
    "month": [7],
    "avg_last_3": [900],
    "growth_rate": [0.04]
})

predicted_spending = model.predict(sample_user)[0]

print("\nPredicted Next Month Spending:", round(predicted_spending, 2))

budget_target = 1600

if predicted_spending > budget_target:
    print("Recommendation: You may exceed your budget next month.")
else:
    print("Recommendation: You are likely within your budget.")
