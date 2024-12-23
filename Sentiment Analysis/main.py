import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Sentiment Analysis/sentiments.txt", sep=";", names=["text","emotion"])
print(data["emotion"].value_counts())
data["emotion"].replace({"joy": 1, "love": 1, "surprise": 1, "sadness": 0, "anger": 0, "fear": 0}, inplace=True)
print(data["emotion"].value_counts())

x_train, y_train, x_test, y_test = train_test_split(data["text"], data["emotion"], train_size=0.7, random_state=50)

# nltk.download('stopwords')
# nltk.download('wordnet')
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
objLemmatizer = WordNetLemmatizer()

def textTransform(texts):
  corpus = []
  for text in texts:
    normal = re.sub('[^a-zA-Z]', ' ', text)
    normal = normal.lower()
    normal = normal.split(' ')
    lemma = []

    for word in normal:
      # if word not in set(stopwords.words('english')):
      if word not in nltk.corpus.stopwords
        lemma.append(objLemmatizer.lemmatize(word))
    corpus.append(' '.join(lemma))
  return corpus

print('corpus')
processedX = textTransform(x_train)
print(processedX[:9])
