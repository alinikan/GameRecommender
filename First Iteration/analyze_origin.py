import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


# Function to convert prices to numerical values
def convert_price(price):
    try:
        return float(price.replace('$', ''))
    except:
        return np.nan


# Function to extract release year from release_date
def extract_year(release_date):
    try:
        return int(str(release_date)[-4:])
    except:
        return np.nan


def analyze_origin_data(df_all):
    # Create directory for outputs if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Preprocessing and feature extraction
    df_all['genre'] = df_all['genre'].str.strip().str.split(',')  # Clean up the genre field
    df_all['popular_tags'] = df_all['popular_tags'].str.strip().str.split(',')  # Clean up the popular_tags field
    df_all['original_price'] = df_all['original_price'].apply(
        convert_price)  # Convert original prices to numerical values
    df_all['discount_price'] = df_all['discount_price'].apply(
        convert_price)  # Convert discount prices to numerical values
    df_all['release_year'] = df_all['release_date'].apply(extract_year)  # Extract release year
    df_all['description_length'] = df_all['game_description'].str.len()  # Create a new feature for description length
    df_all['all_review_score'] = df_all['all_reviews'].str.extract(r'(\d+)%').astype(float)  # Extract all review scores
    df_all['recent_review_score'] = df_all['recent_reviews'].str.extract(r'(\d+)%').astype(
        float)  # Extract recent review scores
    df_all['recent_review_total'] = df_all['recent_reviews'].str.extract(r'of the (\d+)').astype(
        float)  # Extract total number of recent reviews
    df_all['all_review_total'] = df_all['all_reviews'].str.extract(r'of the (\d+)').astype(
        float)  # Extract total number of all reviews

    # Create exploded DataFrame for genre and popular tags
    df_exploded_genre = df_all.explode('genre')
    df_exploded_tags = df_all.explode('popular_tags')

    # Analysis

    # Analyze game genres
    genre_counts = df_exploded_genre['genre'].value_counts()  # Count the number of games in each genre
    plt.figure(figsize=(10, 6))
    genre_counts[:20].plot(kind='bar')  # Plot the 20 most common genres
    plt.title('Top 20 Most Common Game Genres', fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel('Number of Games', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('outputs/Analyze_game_genres.png')

    # Analyze popular tags
    tag_counts = df_exploded_tags['popular_tags'].value_counts()  # Count the number of occurrences of each tag
    plt.figure(figsize=(10, 6))
    tag_counts[:20].plot(kind='bar')  # Plot the 20 most common tags
    plt.title('Top 20 Most Common Tags')
    plt.xlabel('Tag')
    plt.ylabel('Number of Occurrences')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('outputs/Analyze_popular_tags.png')

    # Analyze review scores and total number of reviews
    correlation_matrix = df_all[
        ['recent_review_score', 'recent_review_total', 'all_review_score', 'all_review_total', 'original_price',
         'discount_price']].corr()  # Calculate correlation between review scores, total reviews, and prices
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Review Scores, Total Reviews, and Prices')
    # Adjust subplot parameters to make room for the x-axis labels
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('outputs/Analyze_reviews.png')

    # Analyze game prices
    plt.figure(figsize=(10, 6))
    df_all['original_price'].plot(kind='hist', bins=50, range=(0, 100),
                                  edgecolor='black')  # Plot histogram of original prices
    plt.title('Distribution of Original Prices')
    plt.xlabel('Original Price')
    plt.savefig('outputs/Analyze_prices.png')

    plt.figure(figsize=(10, 6))
    df_all['discount_price'].plot(kind='hist', bins=50, range=(0, 100),
                                  edgecolor='black')  # Plot histogram of discount prices
    plt.title('Distribution of Discount Prices')
    plt.xlabel('Discount Price')
    plt.savefig('outputs/Analyze_discounts.png')

    # Analyze developers
    developer_counts = df_all['developer'].value_counts()  # Count the number of games from each developer
    plt.figure(figsize=(10, 6))
    developer_counts[:20].plot(kind='bar')  # Plot the 20 most prolific developers
    plt.title('Top 20 Most Prolific Developers')
    plt.xlabel('Developer')
    plt.ylabel('Number of Games')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('outputs/Analyze_developers.png')

    # Analyze publishers
    publisher_counts = df_all['publisher'].value_counts()  # Count the number of games from each publisher
    plt.figure(figsize=(10, 6))
    publisher_counts[:20].plot(kind='bar')  # Plot the 20 most prolific publishers
    plt.title('Top 20 Most Prolific Publishers')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Games')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('outputs/Analyze_publishers.png')

    # Analyze release years
    year_counts = df_all['release_year'].value_counts().sort_index()  # Count the number of games released each year

    plt.figure(figsize=(12, 6))
    year_counts[year_counts.index >= 1990].plot(
        kind='line')  # Plot the number of games released each year from 1990 onwards
    plt.title('Number of Games Released Each Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Games')
    plt.grid()
    plt.savefig('outputs/Analyze_release_years.png')

    # Analyze the impact of the pandemic on the game industry
    pandemic_years = year_counts[(year_counts.index >= 2018) & (year_counts.index <= 2022)]
    plt.figure(figsize=(12, 6))
    pandemic_years.plot(kind='line')  # Plot the number of games released each year during the pandemic
    plt.title('Number of Games Released Each Year During the Pandemic')
    plt.xlabel('Year')
    plt.ylabel('Number of Games')
    plt.grid()
    plt.savefig('outputs/Analyze_pandemic.png')

    # Analyze tags and review scores
    tag_review_scores = df_exploded_tags.groupby('popular_tags')[
        'all_review_score'].mean()  # Group by tag and compute average review score
    print(tag_review_scores.nlargest(10))  # Print the tags with the highest average review scores
    print(tag_review_scores.nsmallest(10))  # Print the tags with the lowest average review scores

    # Analyze description length and review score
    correlation = df_all[['description_length',
                          'all_review_score']].corr()  # Compute correlation between description length and review score
    print(correlation)
