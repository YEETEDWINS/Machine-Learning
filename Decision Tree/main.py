import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sb

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('/Users/edwinkcijo/Desktop/just for india/Machine Learning/Decision Tree/car.csv')
data.columns = ("Sales", "Maintenance", "Doors", "Persons", "Boots", "Safety", "Class")

print(data.head())

lblenc = LabelEncoder()
data["Sales"] = lblenc.fit_transform(data["Sales"])
data["Maintenance"] = lblenc.fit_transform(data["Maintenance"])
data["Boots"] = lblenc.fit_transform(data["Boots"])
data["Safety"] = lblenc.fit_transform(data["Safety"])
data["Class"] = lblenc.fit_transform(data["Class"])

x = data[["Sales", "Maintenance", "Boots", "Safety"]]
y = data[["Class"]]

train_x,  test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state= 3)
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)
matrix = confusion_matrix(test_y, y_pred)

sb.heatmap(matrix, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()