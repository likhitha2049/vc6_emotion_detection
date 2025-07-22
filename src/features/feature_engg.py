import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Load processed training and testing data, dropping rows with missing 'content'
train_data = pd.read_csv("data/processed/train.csv").dropna(subset=['content'])
test_data = pd.read_csv("data/processed/test.csv").dropna(subset=['content'])

# Extract features and labels from train and test datasets
X_train = train_data['content'].values
y_train = train_data['sentiment'].values
X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Initialize Bag-of-Words vectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer on training data and transform both train and test data
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Convert the feature vectors to DataFrames for easier handling
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['sentiment'] = y_train  # Add label column to training DataFrame

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['sentiment'] = y_test    # Add label column to testing DataFrame

# Save the transformed data to interim CSV files
train_df.to_csv("data/interim/train_bow.csv", index=False)
test_df.to_csv("data/interim/test_bow.csv", index=False)