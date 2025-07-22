import pandas as pd
import pickle
import json
import logging
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os
# Configure logging
logging.basicConfig(
    filename='logs/model_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def load_model(model_path: str) -> Any:
    """Load a trained model from disk."""
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test data from CSV."""
    try:
        test_data = pd.read_csv(test_path)
        logging.info(f"Test data loaded from {test_path}")
        return test_data
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

def split_features_labels(df: pd.DataFrame) -> Tuple[Any, Any]:
    """Split DataFrame into features and labels."""
    try:
        X_test = df.drop(columns=['sentiment']).values
        y_test = df['sentiment'].values
        logging.info("Split test data into features and labels.")
        return X_test, y_test
    except Exception as e:
        logging.error(f"Error splitting features and labels: {e}")
        raise

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        logging.info("Model evaluation metrics calculated.")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        with open(path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        logging.info(f"Evaluation metrics saved to {path}")
    except Exception as e:
        logging.error(f"Error saving evaluation metrics: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model evaluation."""
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")
        X_test, y_test = split_features_labels(test_data)
        metrics = evaluate_model(model, X_test, y_test)
        os.makedirs("reports", exist_ok=True)
        save_metrics(metrics, "reports/model_evaluation.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Critical error in model evaluation pipeline: {e}")
        raise

if __name__ == "__main__":
    main()