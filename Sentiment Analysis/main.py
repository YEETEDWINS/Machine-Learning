import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import nltk
import re

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Sentiment Analysis/sentiments.txt", sep=";", names=["text","emotion"])
print(data["emotion"].value_counts())
data["emotion"].replace({"joy": 1, "love": 1, "surprise": 1, "sadness": 0, "anger": 0, "fear": 0}, inplace=True)
print(data["emotion"].value_counts())

x_train, x_test, y_train, y_test = train_test_split(data["text"], data["emotion"], train_size=0.7, random_state=50)

import ssl

try:
  _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
  pass
else:
  ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
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
      if word not in set(stopwords.words('english')):
      # if word not in nltk.corpus.stopwords:
        lemma.append(objLemmatizer.lemmatize(word))
    corpus.append(' '.join(lemma))
  return corpus

print('corpus')
processedtrain = textTransform(x_train)
print(processedtrain[:9])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2))
x_train = cv.fit_transform(processedtrain)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

processedtest = cv.transform(textTransform(x_test))
y_pred = rfc.predict(processedtest)
print(y_pred)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print(classification_report(y_test, y_pred))

def identify(statement):
  lemmatext = textTransform(statement)
  datatext = cv.transform(lemmatext)
  prediction = rfc.predict(datatext)
  # for value in range(len(prediction)):
  #   if prediction[value] == 1:
  #     prediction[value] = "+"
  #   elif prediction[value] == 0:
  #     prediction[value] = "-"
  print(prediction)

identify(["I love cheeseburgers","School... It is a terrible place, a nightmare to say the least","Winning is a lot of fun","To kill two birds with one stone makes tasks much easier", "I like eating peanut protein bars"])