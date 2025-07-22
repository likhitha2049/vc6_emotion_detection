import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any, Callable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    filename='logs/data_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logging.error(f"Lemmatization error: {e}")
        return text

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in str(text).split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logging.error(f"Stop word removal error: {e}")
        return text

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logging.error(f"Number removal error: {e}")
        return text

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logging.error(f"Lowercase conversion error: {e}")
        return text

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        return " ".join(text.split()).strip()
    except Exception as e:
        logging.error(f"Punctuation removal error: {e}")
        return text

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"URL removal error: {e}")
        return text

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Set text to NaN if sentence has fewer than 3 words."""
    try:
        df['content'] = df['content'].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
        logging.info("Removed small sentences from DataFrame.")
        return df
    except Exception as e:
        logging.error(f"Small sentence removal error: {e}")
        return df

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the 'content' column of the DataFrame."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Normalized text in DataFrame.")
        return df
    except Exception as e:
        logging.error(f"Text normalization error: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    """Apply all preprocessing steps to a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Sentence normalization error: {e}")
        return sentence

def process_and_save_data(train_path: str, test_path: str, processed_dir: str) -> None:
    """Load, preprocess, and save train and test data."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Loaded raw train and test data.")

        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        logging.info("Applied normalization to train and test data.")

        train_data = remove_small_sentences(train_data)
        test_data = remove_small_sentences(test_data)
        logging.info("Removed small sentences from train and test data.")

        os.makedirs(processed_dir, exist_ok=True)
        train_data.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
        logging.info("Saved processed train and test data.")
    except Exception as e:
        logging.critical(f"Critical error in data preprocessing pipeline: {e}")
        raise

if __name__ == "__main__":
    process_and_save_data("data/raw/train.csv", "data/raw/test.csv", "data/processed")