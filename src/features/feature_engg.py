import numpy as np
import pandas as pd
import os
import logging
import yaml
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(
    filename='logs/feature_engg.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test data, dropping rows with missing 'content'."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info("Loaded processed train and test data.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def extract_features_and_labels(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and labels from train and test DataFrames."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        logging.info("Extracted features and labels from data.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error extracting features and labels: {e}")
        raise

def vectorize_data(
    X_train: np.ndarray, X_test: np.ndarray, max_features: int
) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """Vectorize text data using Bag-of-Words."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Vectorized train and test data using Bag-of-Words.")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Error during vectorization: {e}")
        raise

def save_vectorized_data(
    X_bow, y, path: str
) -> None:
    """Save vectorized data and labels to CSV."""
    try:
        df = pd.DataFrame(X_bow.toarray())
        df['sentiment'] = y
        df.to_csv(path, index=False)
        logging.info(f"Saved vectorized data to {path}")
    except Exception as e:
        logging.error(f"Error saving vectorized data to {path}: {e}")
        raise

def main() -> None:
    """Main function to orchestrate feature engineering."""
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engg']['max_features']

        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train, y_train, X_test, y_test = extract_features_and_labels(train_data, test_data)
        X_train_bow, X_test_bow, _ = vectorize_data(X_train, X_test, max_features)

        os.makedirs("data/interim", exist_ok=True)
        save_vectorized_data(X_train_bow, y_train, "data/interim/train_bow.csv")
        save_vectorized_data(X_test_bow, y_test, "data/interim/test_bow.csv")
        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Critical error in feature engineering pipeline: {e}")
        raise

if __name__ == "__main__":
    main()