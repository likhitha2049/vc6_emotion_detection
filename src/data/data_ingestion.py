import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Read the dataset directly from a GitHub URL
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# Drop the 'tweet_id' column as it's not needed for analysis
df.drop(columns=['tweet_id'], inplace=True)

# Filter the DataFrame to include only 'happiness' and 'sadness' sentiments
final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

# Encode 'happiness' as 1 and 'sadness' as 0 for binary classification
final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})


# Split the data into training and testing sets (80% train, 20% test)
df_test, df_train = train_test_split(final_df, test_size=0.2, random_state=42)

# Create the directory to store raw data if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Save the training and testing data to CSV files
df_test.to_csv('data/raw/test.csv', index=False)
df_train.to_csv('data/raw/train.csv', index=False)
