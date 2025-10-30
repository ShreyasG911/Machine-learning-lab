# Import necessary libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_wine 
import matplotlib.pyplot as plt 
 
# Load dataset 
data = load_wine() 
X = pd.DataFrame(data.data, columns=data.feature_names) 
y = data.target 
 
# Split dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Initialize base estimator and AdaBoost 
base_estimator = DecisionTreeClassifier(max_depth=1) 
model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, 
learning_rate=0.8, random_state=42) 
 
# Train the model 
model.fit(X_train, y_train) 
 
# Predictions 
y_pred = model.predict(X_test) 
 
# Evaluate performance 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
 
# Feature Importance 
plt.figure(figsize=(10, 5)) 
plt.barh(X.columns, model.feature_importances_) 
plt.title("Feature Importance (AdaBoost)") 
plt.xlabel("Importance") 
plt.ylabel("Feature") 
plt.show() 
