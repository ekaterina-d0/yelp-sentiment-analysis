# Required packages for the data set split
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle # To save your model for later
from sklearn.metrics import confusion_matrix, accuracy_score
from time import time

# Required packages for logistic regression
from sklearn.linear_model import LogisticRegression


# Split training and testing/validation sets
df = pd.read_csv('yelpreviews_clean_final.csv')
train, test = train_test_split(df, test_size=0.20, random_state=42) # split the given set into training and test sets
tv = TfidfVectorizer()
article_train = tv.fit_transform(train['text'])
article_test = tv.transform(test['text'])

# logistic regression

lr = LogisticRegression(max_iter=200)  # Make sure the maximum number of iterations to be a number of reasonable values
lr.fit(article_train, train['stars'])
article_predict = lr.predict(article_test)
cm = confusion_matrix(article_predict, test['stars'])
print(cm)
print('Logistic Regression accuracy: ' + str(accuracy_score(article_predict, test['stars'])))

# Add the predictions to the test DataFrame
test['predicted_stars'] = article_predict

# Save the updated DataFrame to a new CSV file
test.to_csv('yelp_reviews_with_predictions.csv', index=False)

print("CSV file with predictions saved successfully.")
