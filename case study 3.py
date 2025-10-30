# House Price Prediction using Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Data Loading and Preparation (Synthetic/Example Data Structure)
# Assume 'housing_data.csv' has columns: 'Price' (Target), 'SqFt', 'Bedrooms', 'Neighborhood', 'YearBuilt'
data = pd.read_csv('housing_data.csv')

# Define features and target
X = data.drop('Price', axis=1)
y = data['Price']

# Identify column types
numerical_features = ['SqFt', 'Bedrooms', 'YearBuilt']
categorical_features = ['Neighborhood', 'Style'] # Example added

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
])

# Step 2: Create Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
])

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Square Error (RMSE): ${rmse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Step 6: Save Model Pipeline
joblib.dump(model, 'price_predictor_pipeline.pkl')
