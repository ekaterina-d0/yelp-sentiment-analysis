# 📝 Yelp Review Sentiment Analysis

This project analyzes Yelp restaurant reviews to classify them into **positive**, **neutral**, or **negative** sentiment categories using natural language processing (NLP) techniques and the VADER sentiment analysis tool.

## 🎯 Objective
To explore patterns in customer feedback and help businesses better understand their reputation and customer satisfaction levels based on review sentiment.

## 🧠 Methods
- **Data Collection**: Yelp restaurant reviews collected via Kaggle dataset.
- **Preprocessing**: Text cleaning, normalization, and tokenization.
- **Sentiment Scoring**: Used VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment analysis tool from NLTK, to score sentiment polarity.
- **Labeling**: Classified reviews as positive, neutral, or negative based on compound scores.

## 📊 Key Findings
- Most Yelp reviews are **overwhelmingly positive**, with fewer neutral and negative entries.
- **Longer reviews** tend to skew more positive, possibly indicating detailed and thoughtful feedback.
- Frequent phrases and topics in negative reviews often point to **service issues or inconsistent quality**.

## 🛠️ Tools & Libraries
- Python, pandas, NLTK, VADER, seaborn, matplotlib, Jupyter Notebook

## 📁 Files
- `sentiment.py`: Main notebook
- `cleaning.py`: Cleaning process
- `yelpreviews-cleanv1.csv`: The original dataset with minimal manual cleaning

## 📌 Next Steps
- Expand the model with additional sentiment classifiers (e.g., TextBlob, BERT)
- Analyze trends across cities or cuisine types
- Incorporate star ratings to cross-reference with sentiment polarity
