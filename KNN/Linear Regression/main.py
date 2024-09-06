import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as mp
import numpy as np
import sklearn.linear_model as sk

x = [10, 20,40, 80,100, 192, 192, 293, 292, 923]
y= [8, 26, 32, 65, 183, 293, 93, 8843,389, 875]

def calcMean(li):
  result = ""
  length = len(li)
  for i in range(length):
    if i == 0:
      val1 = li[i]
      val2 = 0
      result = val1+val2
    else:
      val = li[i]
      result+=val
  result/=length
  return result

# SHOULD BE: 214.2
meanx = calcMean(x)
meany = calcMean(y)

# (sum((Xi - meanx) * (Yi - meany)))/(sum(Xi - meanx)**2)

def calcSlope(lix, liy, mx, my):
  if len(lix) != len(liy):
    print("INVALID INPUT")
    return
  length = len(lix)
  num = 0
  den = 0
  for i in range(length):
    num += ((lix[i] - mx) * (liy[i] - my))
    den += ((lix[i] - mx) ** 2)
  return num/den

def calcIntercept(slope, mx, my):
  return my - (slope * mx)

def splitList(li):
  result = []
  temp_list = []
  for i in li:
    temp_list.append(i)
    result.append(temp_list)
    temp_list = []
  return result

X = np.array(splitList(x))
Y = np.array(splitList(y))
reg = sk.fit(X, Y)

print("m =", reg.coef_)
print("c =", reg.intercept_)

# print(calcIntercept(calcSlope(x, y, meanx, meany), meanx, meany))