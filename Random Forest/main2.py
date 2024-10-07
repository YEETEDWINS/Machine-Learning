import pandas as pd
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Random Forest/student-mat.csv")

columns = list(data.columns)
columns = columns[:-1]
print(columns)

print(data.isna().sum())

x = data[columns]
y = data["G3"]

lblenc = LabelEncoder()
print(data.info())
encodable = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]

for i in encodable:
  x[i] = lblenc.fit_transform(x[i])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=5)
Regressor = RandomForestRegressor()
Regressor.fit(xtrain, ytrain)
predicted = Regressor.predict(xtest)

error = mean_absolute_error(ytest, predicted)
print(error)