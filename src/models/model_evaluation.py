from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import pickle
import json

# Load the trained Random Forest model from disk
with open("models/random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the Bag-of-Words test data
test_data = pd.read_csv("data/interim/test_bow.csv")

# Separate features and labels from test data
X_test = test_data.drop(columns=['sentiment']).values
y_test = test_data['sentiment'].values

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics: accuracy, precision, recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate ROC AUC score using predicted probabilities
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


# Store all evaluation metrics in a dictionary
evaluation_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "roc_auc": roc_auc
}

# Save evaluation metrics to a JSON file for later reference
with open("reports/model_evaluation.json", "w") as metrics_file:
    json.dump(evaluation_metrics, metrics_file, indent=4)