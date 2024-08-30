import numpy as np
import matplotlib as mpl
import sklearn as sk
import pandas as pd
import sklearn.datasets as ds

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
boston = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Poly Regression/boston_house_prices.csv")

X = boston[["LSTAT", "RM"]]
Y = boston["MEDV"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
model = LinearRegression()
model.fit(X_train, Y_train)

Y_test_predict = model.predict(X_test)
rmse_linear_model = (np.sqrt(mean_squared_error(Y_test, Y_test_predict)))
print(rmse_linear_model)

poly_feature = PolynomialFeatures(degree=2)
X_train_poly = poly_feature.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)

X_test_poly = poly_feature.fit_transform(X_test)
Y_test_predict_poly = poly_model.predict(X_test_poly)
rmse_poly_model = (np.sqrt(mean_squared_error(Y_test, Y_test_predict_poly)))
print(rmse_poly_model)

print("")
print(Y_test_predict)
print(Y_test_predict_poly)