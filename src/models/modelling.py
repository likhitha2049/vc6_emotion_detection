import numpy as np
import pandas as pd
import pickle
import logging
import yaml
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
import os
# Configure logging
logging.basicConfig(
    filename='logs/modelling.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def load_params(params_path: str) -> dict:
    """Load model parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def load_train_data(path: str) -> pd.DataFrame:
    """Load Bag-of-Words training data from CSV."""
    try:
        train_data = pd.read_csv(path)
        logging.info(f"Training data loaded from {path}")
        return train_data
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise

def split_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Split DataFrame into features and labels."""
    try:
        X = df.drop(columns=['sentiment']).values
        y = df['sentiment'].values
        logging.info("Split training data into features and labels.")
        return X, y
    except Exception as e:
        logging.error(f"Error splitting features and labels: {e}")
        raise

def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        logging.info("Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: RandomForestClassifier, path: str) -> None:
    """Save the trained model to disk using pickle."""
    try:
        with open(path, "wb") as model_file:
            pickle.dump(model, model_file)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model training."""
    try:
        params = load_params('params.yaml')
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']

        train_data = load_train_data("data/interim/train_bow.csv")
        X_train, y_train = split_features_labels(train_data)
        model = train_model(X_train, y_train, n_estimators, max_depth)
        os.makedirs("models", exist_ok=True)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Critical error in model training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()