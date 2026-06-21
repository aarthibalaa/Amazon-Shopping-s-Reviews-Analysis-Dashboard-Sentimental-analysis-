#importing libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load data
df = pd.read_excel(r'C:\Users\user\Downloads\amazon_review_test.xlsx', engine='openpyxl')
print(df.columns)
df.rename(columns={'Review': 'text'}, inplace=True)

# Step 1: Clean text
def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

df['text'] = df['text'].apply(data_processing)
df = df.drop_duplicates('text')

# Step 2: Apply stemming
def stemming(data):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in word_tokenize(data)]
    return " ".join(text)

df['text'] = df['text'].apply(stemming)

# Step 3: Sentiment polarity
def polarity(text):
    return TextBlob(text).sentiment.polarity

df['polarity'] = df['text'].apply(polarity)

# View result
#print(df.head(10))

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

df['sentiment'] = df['polarity'].apply(sentiment)
print(df.head())
df.to_csv('amazon_reviews_cleaned.csv', index=False)