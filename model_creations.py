
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib


file_path = "twitter_training.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "jp797498e/twitter-entity-sentiment-analysis",
  file_path,
 
)

model=make_pipeline(TfidfVectorizer(),MultinomialNB())

df['im getting on borderlands and i will murder you all ,'] = df['im getting on borderlands and i will murder you all ,'].fillna('')
x_train,x_test,y_train,y_test=train_test_split(df['im getting on borderlands and i will murder you all ,'],df['Positive'],test_size=0.3,random_state=42)
model.fit((x_test),(y_test))


joblib.dump(model,'senti.ykl')
