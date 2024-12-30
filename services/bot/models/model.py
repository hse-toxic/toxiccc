from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
import pymorphy3
from nltk import word_tokenize
from nltk.corpus import stopwords
import re


nltk.download('stopwords')
nltk.download('punkt_tab')
morph = pymorphy3.MorphAnalyzer()
ru_stopwords = stopwords.words("russian") + ['это', 'её']

model = load('models/model.joblib')
vectorizer = load('models/vectorizer.joblib')


def transform(text: str):
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        text = re.sub(r"\'s", " ", text)
        # text = re.sub('\W', ' ', text)
        # text = re.sub('\s+', ' ', text)
        # text = text.strip(' ')
        return text

    text = clean_text(text)
    text = " ".join(
        filter(
            lambda word: word not in ru_stopwords,
            map(
                lambda word: morph.parse(word)[0].normal_form,
                word_tokenize(text)
            )
        )
    )

    return vectorizer.transform([text])


def predict(text: str):
    return model.predict(transform(text))[0]
