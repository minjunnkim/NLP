import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch
from transformers import pipeline
import spacy

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

# Custom lexicon
custom_lexicon = {
    'amazing': 1.0,
    'great': 0.9,
    'good': 0.8,
    'bad': -0.8,
    'terrible': -0.9,
    'horrible': -1.0,
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

def analyze_sentiment(text):
    preprocessed_text = preprocess_text(text)
    
    # Custom lexicon scoring
    custom_score = sum([custom_lexicon.get(w, 0) for w in preprocessed_text.split()])
    
    # SentimentIntensityAnalyzer scoring
    sia_score = sia.polarity_scores(preprocessed_text)['compound']
    
    # Ensemble scoring
    sentiment_score = (custom_score + sia_score) / 2
    
    if sentiment_score > 0.3:
        return 'Positive'
    elif sentiment_score < -0.3:
        return 'Negative'
    else:
        return 'Neutral'
    
# Ensemble learning with aspect-based and overall sentiment analysis
def ensemble_analyze_sentiment(text):
    preprocessed_text = preprocess_text(text)
    
    # Custom lexicon scoring
    custom_score = sum([custom_lexicon.get(w, 0) for w in preprocessed_text.split()])
    
    # SentimentIntensityAnalyzer scoring
    sia_score = sia.polarity_scores(preprocessed_text)['compound']
    
    # Aspect-based sentiment analysis for "quality"
    quality_score = aspect_based_sentiment_analysis(text, "quality")
    
    # Aspect-based sentiment analysis for "price"
    price_score = aspect_based_sentiment_analysis(text, "price")
    
    # Overall sentiment analysis
    overall_score = overall_sentiment_analysis(text)
    
    # Ensemble scoring
    sentiment_score = (custom_score + sia_score + quality_score + price_score + overall_score) / 5
    
    # Error analysis
    if custom_score * sia_score < 0:
        # If the custom lexicon and SIA have opposite sentiment polarity, use SIA score
        sentiment_score = sia_score
    elif abs(custom_score - sia_score) > 0.5:
        # If the difference between custom lexicon and SIA scores is greater than 0.5, use SIA score
        sentiment_score = sia_score
    
    return sentiment_score  

# # Machine learning algorithm
# def train_model():
#     with open('trainModel.pickle', 'rb') as f:
#         stmts = pickle.load(f)
#     X = [preprocess_text(stmt) for stmt, label in stmts]
#     y = [label for review, label in stmts]
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(X)
#     clf = MultinomialNB()
#     clf.fit(X, y)
#     return vectorizer, clf

# Load the pre-trained sentiment analysis model
model = pipeline('sentiment-analysis')

nlp = spacy.load("en_core_web_sm")

# Define a function to perform aspect-based sentiment analysis
def aspect_based_sentiment_analysis(text, aspect):
    # Split the text into sentences
    sentences = text.split('. ')
    
    # Define a list to store the sentiment scores for each sentence mentioning the aspect
    scores = []
    
    for sentence in sentences:
        # If the aspect is mentioned in the sentence, get its sentiment score
        if aspect in sentence.lower():
            # Apply dependency parsing and named entity recognition to the sentence
            doc = nlp(sentence)
            
            # Find the relevant noun or verb associated with the aspect
            relevant_token = None
            for token in doc:
                if token.text.lower() == aspect:
                    relevant_token = token
                    break
            
            # If no relevant token was found, move on to the next sentence
            if relevant_token is None:
                continue
            
            # Determine the sentiment of the relevant token's subtree
            subtree_sentiment = 0
            for child in relevant_token.subtree:
                if child.dep_ in ['amod', 'advmod']:
                    # If the child is an adjective or adverb modifying the relevant token, get its sentiment score
                    result = model(child.text)
                    score = result[0]['score']
                    subtree_sentiment += score
                elif child.dep_ == 'neg':
                    # If the child is a negation, invert the sentiment score of the relevant token's subtree
                    subtree_sentiment *= -1
            
            # If no sentiment was found in the subtree, move on to the next sentence
            if subtree_sentiment == 0:
                continue
            
            # Add the sentiment score of the relevant token's subtree to the list of scores
            scores.append(subtree_sentiment)
    
    # If no sentence mentions the aspect, return a neutral sentiment score
    if not scores:
        return 0.0
    
    # Calculate the average sentiment score for the aspect
    avg_score = sum(scores) / len(scores)
    
    return avg_score

# Define a function to perform sentiment analysis on the overall text
def overall_sentiment_analysis(text):
    result = model(text)
    score = result[0]['score']
    return score

def predict_sentiment(text, vectorizer, clf):
    preprocessed_text = preprocess_text(text)
    X = vectorizer.transform([preprocessed_text])
    y = clf.predict(X)
    if y == 0:
        return 'Negative'
    elif y == 1:
        return 'Neutral'
    elif y == 2:
        return 'Positive'

while True:
    text = input("Enter some text to analyze sentiment (Type q to quit): ")

    if text.lower() == "q":
        break

    else:
        try:
            sentiment_score = ensemble_analyze_sentiment(text)
            print(sentiment_score)

            if sentiment_score > 0.3:
                print('Positive')
            elif sentiment_score < -0.3:
                print('Negative')
            else:
                print('Neutral')
        except:
            print("An error occurred during sentiment analysis. Please try again.")