import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/KNN/data.csv")

X = data[["sepal_length", "sepal_width", "petal_length","petal_length","petal_width"]]
Y = data[["species"]]

print(data.head)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=5)
StandardScaler = StandardScaler()
StandardScaler.fit_transform(train_x)

labelenc = LabelEncoder()
labelenc.fit_transform(train_y)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_x, train_y)
StandardScaler.transform(test_x)

y_pred = classifier.predict(test_x)
labelenc.transform(test_y)

matrix = confusion_matrix(test_y, y_pred)

print(classification_report(test_y, y_pred))

sb.heatmap(matrix, annot=True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()