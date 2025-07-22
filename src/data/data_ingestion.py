import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    filename='logs/data_ingestion.log',
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

def fetch_dataset(url: str) -> pd.DataFrame:
    """Fetch dataset from a given URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Dataset loaded from {url}")
        return df
    except Exception as e:
        logging.error(f"Error fetching dataset: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset for binary classification."""
    try:
        # Drop unnecessary columns
        df = df.drop(columns=['tweet_id'])
        logging.info("Dropped 'tweet_id' column.")

        # Filter for 'happiness' and 'sadness'
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        logging.info("Filtered for 'happiness' and 'sadness' sentiments.")

        # Encode sentiments
        df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0}).astype(int)
        logging.info("Encoded sentiments for binary classification.")

        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets."""
    try:
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train and test sets with test_size={test_size}")
        return df_train, df_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    try:
        df.to_csv(path, index=False)
        logging.info(f"Data saved to {path}")
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']

        df = fetch_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        df_train, df_test = split_data(final_df, test_size)

        os.makedirs('data/raw', exist_ok=True)
        logging.info("Ensured 'data/raw' directory exists.")

        save_data(df_train, 'data/raw/train.csv')
        save_data(df_test, 'data/raw/test.csv')
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Critical error in data ingestion pipeline: {e}")
        raise

if __name__ ==  "__main__":
    main()