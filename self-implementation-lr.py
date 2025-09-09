import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('Salary Data.csv')
df = df[['Years of Experience', 'Salary']].dropna()

X = df['Years of Experience'].values
y = df['Salary'].values


# Initialize parameters
w = 0.0  # slope
b = 0.0  # intercept

# Hyperparameters
lr = 0.0001   # learning rate
epochs = 300

# Store loss history
loss_history = []

# Training loop
for epoch in range(epochs):
    # Predictions
    y_pred = w * X + b

    # Loss (Mean Squared Error)
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)

    # Gradients
    dw = (2/len(X)) * np.sum((y_pred - y) * X)
    db = (2/len(X)) * np.sum(y_pred - y)

    # Update weights
    w -= lr * dw
    b -= lr * db

    # Print progress every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.2f}, w: {w:.4f}, b: {b:.4f}")

# ---- Final Model ----
print("\nFinal parameters:")
print(f"Slope (w): {w}, Intercept (b): {b}")


# ---- Calculate R^2 ----
y_pred_final = w * X + b
ss_res = np.sum((y - y_pred_final) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"R^2 score: {r2:.4f}")

# ---- Plot results ----
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, w * X + b, color="green", label="Fitted Line")
plt.xlabel("Years of Experience (normalized)")
plt.ylabel("Salary")
plt.title("Gradient Descent Linear Regression")
plt.legend()
plt.show()

# ---- Plot loss over epochs ----
#plt.plot(loss_history)
#plt.xlabel("Epoch")
#plt.ylabel("MSE Loss")
#plt.title("Loss Curve")
#