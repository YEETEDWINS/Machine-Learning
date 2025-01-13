import sklearn as sk
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Heart Disease Prediction/heart.csv")

y = data["target"]

x = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]

scalingObject = MinMaxScaler()
x = scalingObject.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=5)

obj = svm.SVC()
obj.fit(x_train, y_train)

y_pred = obj.predict(x_test)

print("Only SVM'ed")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
report = classification_report(y_test, y_pred)
print(report)

FeatureObj = PCA(10)
Smallerx_train = FeatureObj.fit_transform(x_train)
Smallerx_test = FeatureObj.fit_transform(x_test)

obj.fit(Smallerx_train, y_train)
newy_pred = obj.predict(Smallerx_test)

print("PCA'ed values")
newmatrix = confusion_matrix(y_test, newy_pred)
print(newmatrix)
newreport = classification_report(y_test, newy_pred)
print(newreport)
