# Customer Churn Prediction using Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, request, jsonify
import joblib

# Step 1: Data Loading and Preprocessing
data = pd.read_csv('telecom_churn.csv')
data.columns = data.columns.str.replace(' ', '_') # Clean column names
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

# Convert categorical features to numerical
for column in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
               'Contract', 'PaperlessBilling', 'PaymentMethod', 'InternetService']:
    if column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Select features and target
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'gender', 'Partner', 'Dependents']
X = data[features]
y = data['Churn']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 5: Save Model and Scaler
joblib.dump((model, scaler, features), 'churn_predictor.pkl')

# Step 6: Flask Deployment Snippet
app = Flask(__name__)
model, scaler, features = joblib.load('churn_predictor.pkl')

@app.route('/predict_churn', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # The input will be a dictionary matching the feature list
    input_df = pd.DataFrame([data])
    input_scaled = scaler.transform(input_df[features])
    
    prediction = model.predict(input_scaled)[0]
    churn_label = 'High Risk - Churn' if prediction == 1 else 'Low Risk - Ham'
    
    return jsonify({'prediction': churn_label})

if __name__ == '__main__':
    # Running the app on a local port
    # app.run(port=5000, debug=False)
    pass # Code is run in a notebook/script environment
