import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report

sentiment = SentimentIntensityAnalyzer()

file = 'yelpreviews_clean_final.csv'
df = pd.read_csv(file)

df['Sentiment_Compound'] = [sentiment.polarity_scores(row)['compound'] for row in df['text']]
df['Positive_Sentiment'] = [sentiment.polarity_scores(row)['pos'] for row in df['text']]
df['Neutral_Sentiment'] = [sentiment.polarity_scores(row)['neu'] for row in df['text']]
df['Negative_Sentiment'] = [sentiment.polarity_scores(row)['neg'] for row in df['text']]

print(df['Sentiment_Compound'].astype(str) + '  ' + df['text'])
save_file = file.split('.')[0]+'_with_sentiment.csv'
df.to_csv(save_file, index = False)

# Compute the confusion matrix
conf_matrix = confusion_matrix(df['Actual_Sentiment'], df['Predicted_Sentiment'], labels=['positive', 'neutral', 'negative'])
print("Confusion Matrix:\n", conf_matrix)

