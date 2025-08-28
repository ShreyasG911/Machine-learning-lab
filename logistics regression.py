#Roll no CS3155
# predefine dataset import
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# splitting dataset with training and testing
from sklearn.model_selection import train_test_split
# parameters to check accuracy of model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import pandas as pd
import matplotlib.pyplot as plt


# Load the Iris dataset
iris = load_iris() # function to load the data
X = iris.data[:, :2]
y = (iris.target == 0).astype(int)
# satosa to 1 and other to 0
# variable for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
# model fitting
y_pred = model.predict(X_test)

print(f"Accuracy: ",accuracy_score(y_test, y_pred))
print(f"Classification Report: \n",classification_report(y_test, y_pred))
print(f"Confusion Matrix: \n",confusion_matrix(y_test, y_pred))
