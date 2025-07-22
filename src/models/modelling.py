import numpy as np
import pandas as pd
import re
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the Bag-of-Words training data
train_data = pd.read_csv("data/interim/train_bow.csv")

# Separate features and labels
X_train = train_data.drop(columns=['sentiment']).values
y_train = train_data['sentiment'].values

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model to disk using pickle
with open("models/random_forest_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)