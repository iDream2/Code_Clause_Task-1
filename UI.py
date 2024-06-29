import pickle
import streamlit as st
import numpy as np
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained model and TF-IDF vectorizer
pickled_model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Assuming you saved the TF-IDF vectorizer separately

# Define preprocessing functions
def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenization(text):
    tokens = re.split(r'\W+', text)
    return tokens

stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]
    return output

porter_stemmer = PorterStemmer()

def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

wordnet_lemmatizer = WordNetLemmatizer()

def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def preprocess_review(review):
    review = remove_punctuation(review.lower())
    review = tokenization(review)
    review = remove_stopwords(review)
    review = stemming(review)
    review = lemmatizer(review)
    return ' '.join(review)

# Streamlit UI
st.title("Sentiment Analysis of Movie Reviews")

review_input = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review_input:
        processed_review = preprocess_review(review_input)
        review_tfidf = tfidf.transform([processed_review]).toarray()
        prediction = pickled_model.predict(review_tfidf)
        sentiment = 'positive' if prediction == 1 else 'negative'
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to analyze.")


