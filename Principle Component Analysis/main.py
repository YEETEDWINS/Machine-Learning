# Unsupervised Learning
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from matplotlib import pyplot

dataset = datasets.load_breast_cancer()
df = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
target = dataset["target"]

print(df.info())

objscaler = MinMaxScaler()
dataScaled = objscaler.fit_transform(df) # transformed data between 0 and 1
objPCA = PCA(2)
data = objPCA.fit_transform(dataScaled)

print(dataScaled)
print(data)

pyplot.scatter(data[:, 0], data[:, 1], c=target)
pyplot.show()

xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2, random_state=50)
obj = RandomForestClassifier()
obj.fit(xtrain, ytrain)

ypred = obj.predict(xtest)

matrix = confusion_matrix(ytest, ypred)
print(matrix)
print(classification_report(ytest, ypred))
