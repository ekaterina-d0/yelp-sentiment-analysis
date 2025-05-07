import pandas as pd
from nltk.corpus import wordnet
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

file = 'yelpreviews-cleanv1.csv'
df = pd.read_csv(file)

# general overview
print(df.columns)
print(df.head())
print(df.isnull().sum())

# remove repeating characters & lowercase everything
def remove_repeated_characters(text):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')  # regex metacharacters for identifying repeating patterns
    match_substitution = r'\1\2\3'  # we define 3+ repeating characters to be true repeating characters
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    print(text)
    correct_text = text.apply(lambda x: ' '.join([replace(word) for word in x.split(' ')]))
    return correct_text

df['text'] = remove_repeated_characters(df['text'].str.lower())


# remove stopwords
tokenizer = ToktokTokenizer()
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)  # tokenize the provided text
    tokens = [token.strip() for token in tokens]  # list comprehension. Create a list of tokens
    filtered_tokens = [token for token in tokens if token not in stopwords]  # only keeping the non-stopword words
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))


# stemming
ps = PorterStemmer()
def simple_stemmer(text):
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

df['text'] = df['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

df.to_csv('yelpreviews_clean_final.csv', header=True, index=False)