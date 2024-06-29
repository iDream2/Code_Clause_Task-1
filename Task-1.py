import numpy as np
import pandas as pd
import nltk
import string
string.punctuation
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

df = pd.read_csv("./IMDB Dataset.csv")
pd.set_option('display.max_colwidth', None)



#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#storing the puntuation free text
df['review']= df['review'].apply(lambda x:remove_punctuation(x))


#converting to lowercase
df['review']= df['review'].apply(lambda x: x.lower())


#tokenizing the text
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

df['review']= df['review'].apply(lambda x: tokenization(x))

#specifying the stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords[0:21]
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "<br>", "</br>", "a", "an", "the", "of", "to", "are", "is", "they","them" ]

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
df['review']= df['review'].apply(lambda x:remove_stopwords(x))





#Stemming
porter_stemmer = PorterStemmer()

def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
df['review']=df['review'].apply(lambda x: stemming(x))


#Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()


def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
df['review']=df['review'].apply(lambda x:lemmatizer(x))

# Join the tokens back into strings
df['review'] = df['review'].apply(lambda x: ' '.join(x))

# Transform the text data into numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['review']).toarray()
pickle.dump(tfidf, open("tfidf.pkl", "wb")   )


# Encode the sentiment column (assuming the sentiment column contains 'positive' and 'negative' values)
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
y = df['sentiment'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (Logistic Regression)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

pickle.dump(classifier, open("model.pkl" , 'wb' )  )

# Evaluate the model on the test data
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Function to predict sentiment for new reviews
def predict_sentiment(review):
    review = remove_punctuation(review.lower())
    review = tokenization(review)
    review = remove_stopwords(review)
    review = stemming(review)
    review = lemmatizer(review)
    review = ' '.join(review)
    review_tfidf = tfidf.transform([review]).toarray()
    prediction = classifier.predict(review_tfidf)
    return 'positive' if prediction == 1 else 'negative'

# Example usage
new_review = "This movie was fantastic! I really enjoyed it."
print("Sentiment:", predict_sentiment(new_review))
