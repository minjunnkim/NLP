import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')

#MARK -- Text pre-processing
import pandas as pd

df = pd.read_csv('dataset.csv')

#MARK -- Clean the word
import re
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

#MARK -- Sentiment Analysis 
sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return sia.polarity_scores(text)['compound']

df['sentiment_score'] = df['clean_text'].apply(get_sentiment_score)

#MARK -- Visualize
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['sentiment_score'])
plt.show()