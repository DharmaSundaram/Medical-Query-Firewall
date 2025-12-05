# train_classifier.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_PATH = "data/sample_data.csv"
MODEL_PATH = "models/classifier.joblib"
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df['text']
y = df['label']

# Simple pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(X, y)
joblib.dump(pipe, MODEL_PATH)
print("Saved classifier to", MODEL_PATH)
