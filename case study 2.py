# Predictive Maintenance: Failure Classification using ML

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib

# Step 1: Data Loading and Feature Engineering (Simplified)
# Assume 'maintenance_data.csv' has columns: 'machineID', 'datetime', 'sensor_1', ..., 'sensor_N', 'failure_next_24hr' (Target)
data = pd.read_csv('maintenance_data.csv')

# Create an example synthetic feature set for illustration
# In a real scenario, this would involve rolling window features on sensor data (e.g., mean temp over last 24h)
features = ['sensor_1_avg_24h', 'sensor_2_max_24h', 'vibration_std_48h', 'age_of_part']
X = data[features]
y = data['failure_next_24hr'] # 1=Imminent Failure, 0=Normal Operation

# Handle imbalanced data (Crucial for failure prediction, often using SMOTE or class weights)
# This step is omitted in the snippet for brevity but is essential in practice.

# Step 2: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Model Training (Random Forest Classifier for Binary Classification)
# Use class_weight='balanced' to account for rare failure events
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Probability of failure

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 6: Save Model and Scaler
joblib.dump((model, scaler, features), 'pdm_classifier.pkl')
