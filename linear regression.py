# ROLL NO. CS3155
# Assignment 1
# 1.1 Linear Regression

file_path = '/content/drive/My Drive/Colab Notebooks/linear_regression_dataset.csv'

import pandas as pd

# Reading the dataset
df = pd.read_csv(file_path)

# Displaying the dataset
print(df.head())

# Importing necessary libraries
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Splitting the data into X and Y
X = df[['x']]  # X should be a 2D array
y = df['y']

# Creating the Linear Regression model
reg = LinearRegression()

# Fitting the model to the data
reg.fit(X, y)

# Making predictions
y_pred = reg.predict(X)

# Single prediction
y_pred_single = reg.predict([[9]])
print(f"Prediction for x=10: {y_pred_single[0]}")

# Score
score = reg.score(X, y)
print(f"Score: {score}")

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Outputting the model parameters
print(f"Intercept: {reg.intercept_}")
print(f"Coefficient: {reg.coef_}")
