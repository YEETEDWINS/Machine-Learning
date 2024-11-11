import sklearn as sk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Content-Based Recommendation/movies_metadata.csv")

print(data["overview"].head(10))
print(data.shape)

print(data["overview"].isna().sum())
delete = data[data['overview'].isna()].index
data.drop(delete, inplace=True)
print(data.shape)
obj = TfidfVectorizer(stop_words="english")
TfidMatrix = obj.fit_transform(data["overview"])
print(TfidMatrix.shape)

print(TfidMatrix[5000:5010])
print(obj.get_feature_names_out()[2000:2010])
indices = pd.Series(data.index, index=data["title"]).drop_duplicates()
print(indices.head(20))

SIMobj = linear_kernel(TfidMatrix, TfidMatrix)
print(SIMobj[1000:1010])

def Recommend(title):
  index = indices[title]
  simScores = list(enumerate(SIMobj[index]))
  simScores.