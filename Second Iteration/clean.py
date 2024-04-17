import pandas as pd
import re
import string


# Redefining the function to clean and normalize text data without removing stop words
def clean_normalize_text_updated(text):
    """
    Clean and normalize text data without removing stop words:
    - Lowercase the text
    - Remove URLs, HTML tags, and non-alphanumeric characters
    """
    text = text.lower()  # Lowercasing
    text = re.sub(r'http\S+', '', text)  # Removing URLs
    text = re.sub(r'<.*?>', '', text)  # Removing HTML tags
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Removing punctuation
    return text


def preprocess_data_updated(file_path):
    """
    Preprocess the dataset:
    - Load the dataset
    - Apply updated text cleaning and normalization on relevant text columns
    - Drop unnecessary columns
    - Save the cleaned dataset to a new CSV file
    """
    df = pd.read_csv(file_path)

    text_columns = ['desc_snippet', 'all_reviews', 'game_description']
    for col in text_columns:
        df[col] = df[col].astype(str).apply(clean_normalize_text_updated)

    columns_to_drop = ['minimum_requirements', 'recommended_requirements', 'discount_price', 'achievements']
    df.drop(columns=columns_to_drop, inplace=True)

    cleaned_file_path_updated = 'data/final.csv'
    df.to_csv(cleaned_file_path_updated, index=False)

    return cleaned_file_path_updated


# Specify the correct path to your dataset
file_path = '../First Iteration/data/combined.csv'
cleaned_file_path_updated = preprocess_data_updated(file_path)

print(f"Cleaned dataset saved to: {cleaned_file_path_updated}")
