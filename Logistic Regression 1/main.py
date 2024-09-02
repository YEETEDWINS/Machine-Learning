import pandas as pd
import numpy as np
import matplotlib as mpl
import sklearn as sk

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Classifcation report gives the answer to your report, Confusion matrix helps out to find the probability
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Logistic Regression 1/titanic.csv")
print("Missing Values: ", data.isnull().sum().sum())

print("Median of Age Column: ", data["Age"].median(skipna=True))
recdata = (data["Cabin"].isnull().sum())/(data.shape[0])
print("Percent of Missing Records in the Cabin: ", recdata*100)
print("Most Prominent Boarding Point: ", data["Embarked"].value_counts().idxmax())

data.fillna({"Age": 28}, inplace=True)
data.fillna({"Embarked": data["Embarked"].value_counts().idxmax()}, inplace=True)
"""
Older version of Preprocessing

data["Age"].fillna(28, inplace=True)
data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(), inplace=True)
"""

data["Travel Alone"] = np.where((data["SibSp"]+data["Parch"]) > 0, 0, 1)

data.drop("Cabin", axis=1, inplace=True)
data.drop("PassengerId", axis=1, inplace=True)
data.drop("Name", axis=1, inplace=True)
data.drop("SibSp", axis=1, inplace=True)
data.drop("Parch", axis=1, inplace=True)
data.drop("Ticket", axis=1, inplace=True)
print("Missing Values: ", data.isnull().sum().sum())

lblenc = preprocessing.LabelEncoder()
data["Sex"] = lblenc.fit_transform(data["Sex"])
data["Embarked"] = lblenc.fit_transform(data["Embarked"])
print(data.head())

x = data[["Pclass","Sex","Age","Fare","Embarked","Travel Alone"]]
y = data[["Survived"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
lrmodel = LogisticRegression()
lrmodel.fit(x_train, y_train)
y_predict = lrmodel.predict(x_test)
matrix = confusion_matrix(y_test, y_predict)

print(classification_report(y_test, y_predict))