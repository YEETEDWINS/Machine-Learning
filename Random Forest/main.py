import pandas as pd
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import os

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Random Forest/adult_income.csv", sep=", ")

X = data.iloc[0:,0:12]
y = data.iloc[0:,-1]

encoding = ["workclass","education","marital-status","occupation","relationship","race","gender"]
objectencoder = LabelEncoder()
for i in encoding:
  X[i] = objectencoder.fit_transform(X[i])

y = objectencoder.fit_transform(y)
print(X[:5])
print(y[:5])

print(data.info())


xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=5)

Classifier = RandomForestClassifier(n_estimators=100) # default is 100, it is the number of decision trees used
Classifier.fit(xtrain, ytrain)

predicted = Classifier.predict(xtest)
print(predicted)

os.system("clear")
matrix = confusion_matrix(ytest, predicted)
print(classification_report(ytest, predicted))
print(matrix)