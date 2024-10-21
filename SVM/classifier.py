from sklearn import datasets
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

dataset = datasets.load_breast_cancer()
print(dataset.keys())
table = pd.DataFrame(dataset.data, columns= dataset.feature_names)
print(table.head())
print(table.isna().sum())

y = dataset.target33
x_train, x_test, y_train, y_test = train_test_split(table, y)

obj = svm.SVC(kernel='linear')
obj.fit(x_train, y_train)
ypred = obj.predict(x_test)
print(ypred)

print(classification_report(y_test, ypred))
matrix = confusion_matrix(y_test, ypred)
print(matrix)