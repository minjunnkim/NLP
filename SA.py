import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def scoreInterpreter(scores):
    # Determine the overall sentiment
    if scores >= 0.05:
        sentiment = 'Positive'
    elif scores <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment

while True:
    user_input = input("Enter text to analyze sentiment (q to quit): ")
    if user_input.lower() == 'q':
        break
    else:
        sentiment_score = sia.polarity_scores(user_input)['compound']
        print("Sentiment score:", sentiment_score)
        sentiment = scoreInterpreter(sentiment_score)
        print(f"Sentiment: {sentiment}")
        