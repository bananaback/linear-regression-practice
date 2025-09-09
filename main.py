import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Salary_dataset.csv')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

print(df.head())

# Features and target
X = df[['YearsExperience']]
y = df['Salary']

# Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Mean salary (baseline)
mean_salary = y.mean()
print(f"Mean Salary: {mean_salary}")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_all = model.predict(X)       # predictions for all points
y_pred_test = model.predict(X_test) # predictions for test set

# Model parameters
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope}, Intercept: {intercept}")

# R^2 scores
print("Train R^2:", model.score(X_train, y_train))
print("Test R^2:", model.score(X_test, y_test))

# ---- Plot ----
plt.scatter(X, y, color="blue", label="Data points")

# Mean line
plt.axhline(y=mean_salary, color="red", linestyle="--", label="Mean Salary")

# Regression line (sorted X for smooth plotting)
X_sorted = X.sort_values(by="YearsExperience")   # stays DataFrame
y_pred_sorted = model.predict(X_sorted)

plt.plot(X_sorted, y_pred_sorted, color="green", linewidth=2, label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.legend()
plt.show()
