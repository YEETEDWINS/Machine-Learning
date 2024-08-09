import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print("")
print(x)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print("")
print(x)

le = LabelEncoder()
y = le.fit_transform(y)

print("")
print(y)

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=1)
print("")
print(x_tr)
print(x_te)
print(y_tr)
print(y_te)

sc = StandardScaler()
x_tr[:, 3:] = sc.fit_transform(x_tr[:, 3:])
x_te[:, 3:] = sc.transform(x_te[:, 3:])
print("")
print(x_tr)
print(x_te)