import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("HousingData.csv")
data = data.dropna()

print(data.head())


# Separate features and target
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

print("Features:")
print(X.head())

print("Target:")
print(y.head())

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)


from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model using training data
model.fit(X_train, y_train)

print("Model training completed")


# Predict house prices using test data
y_pred = model.predict(X_test)

print("Predicted prices:")
print(y_pred[:10])


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot graph of Actual vs Predicted prices
plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.show()