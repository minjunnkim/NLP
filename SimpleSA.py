import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def get_sentiment(text):
    # Instantiate a sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Calculate the sentiment scores for the text
    scores = sia.polarity_scores(text)

    # Determine the overall sentiment
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment

# Example usage
text = "What the fuck"
sentiment = get_sentiment(text)
print(f"Sentiment: {sentiment}")