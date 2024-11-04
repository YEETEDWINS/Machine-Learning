import sklearn as sk
import pandas as pd

mainData = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Simple Recommendation/movies_metadata.csv")
data = mainData[["title", "popularity", "vote_average", "vote_count"]]

data.dropna(inplace=True)
print(data.isnull().sum())
print(data.info())

"""
Weighted Rating - 
(v/v+m)*R + (m/v+m)*C
v is the number of votes (vote_count)
m is the minimum votes required to be listed in chart
R is the average rating for the movie (vote_average)
C is the mean vote across the whole report
"""

minVotes = data["vote_count"].quantile(0.9)
print(minVotes)

mean = data["vote_average"].mean()
goodData = data.loc[data["vote_count"] >= minVotes]

def calcRating(dataset):
  v = dataset["vote_count"]
  m = minVotes
  R = dataset["vote_average"]
  C = mean
  return (v/v+m)*R + (m/v+m)*C

goodData["Weighted Rating"] = goodData.apply(calcRating, axis=1)
print(goodData.sort_values("Weighted Rating", ascending=False).head(20))