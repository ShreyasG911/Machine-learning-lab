# ROLL NO. CS3155
# 1.2 Polynomial regression

file_path = '/content/drive/My Drive/Colab Notebooks/polynomial_regression_dataset.csv'

import pandas as pd

# Reading the dataset
df = pd.read_csv(file_path)

# Displaying the dataset
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Assuming the dataset has 'x' and 'y' columns
x = df['x'].values.reshape(-1, 1)  # Feature
y = df['y'].values  # Target variable

# Split the data into training and testing sets (optional but recommended)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create a PolynomialFeatures object (degree=3)
degree = 3
poly_features = PolynomialFeatures(degree=degree)

# Transform the data into polynomial features
x_poly_train = poly_features.fit_transform(x_train)

# Create a linear regression model
poly_model = LinearRegression()

# Fit the model to the polynomial features
poly_model.fit(x_poly_train, y_train)

# Predict the values using the polynomial model
x_poly_test = poly_features.transform(x_test)
y_pred = poly_model.predict(x_poly_test)

# Displaying the results
print(f"Polynomial Regression Model (Degree={degree})")
print(f"Intercept: {poly_model.intercept_}")
print(f"Coefficients: {poly_model.coef_}")

# Plotting the original data points and the polynomial regression curve
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Original Data', s=30)
plt.plot(x, poly_model.predict(poly_features.fit_transform(x)), color='red', label=f'Polynomial Regression (Degree={degree})')
plt.title(f"Polynomial Regression (Degree={degree})")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
