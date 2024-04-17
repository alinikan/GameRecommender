import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

free_price_pattern = re.compile(r'(F|f)ree')
numerical_price_pattern = re.compile(r'\$(\d+(\.\d(\d)?)?)')
only_valid_characters_number_pattern = re.compile(r'^[a-zA-Z\s!"$&\'()*+,-.:?_~]+$')


def trim_features(df):
    return df.drop(columns=['url', 'types', 'recent_reviews'])


def filter_prices_games(price):
    price = str(price)

    matched = numerical_price_pattern.search(price)
    if free_price_pattern.search(price):
        # if a game is free, then we return 0 as its price
        return 0
    elif matched:
        # if a game had an acceptable price with a $ unit, then
        # the value of the price is returned
        return float(matched.group(1))
    else:
        # Some game doesn't have free or a price in their price entry
        # for instance, a game had only word 'Demo' as its price
        # we'll return None for those games
        return None


def filter_non_valid_game_names(name):
    name = str(name)
    matches = only_valid_characters_number_pattern.search(name)
    if matches:
        return name  # The game name has no non english or invalid characters such as emojies
    else:
        return None  # if the game


# END
def clean_data(df):
    # Remove 'publisher' column
    df = df.drop(columns=['publisher'])

    # Define a list of columns to check for NaN values
    columns_to_check = ['name', 'desc_snippet', 'release_date', 'developer', 'genre', 'recent_reviews', 'all_reviews',
                        'original_price']

    # Remove any rows with NaN values in the specified columns
    df = df.dropna(subset=columns_to_check)

    # Remove any rows that contains 'bundle' in the 'types' column (Case insensitive).
    df = df[~df['types'].str.contains('bundle', case=False)]

    # Define a list of keywords to remove from 'name' column
    keywords = ['pack', 'soundtrack', 'dlc', 'add-on', 'expansion', 'Additional', 'costume', 'Outfit', 'Graphic',
                'Expendable', 'Song', 'Outfits', 'Bundle', 'Kit']

    # Remove any rows that contains keywords in the 'name' column (Case insensitive).
    df = df[~df['name'].str.contains('|'.join(keywords), case=False)]

    # Remove "About This Game" from 'game_description' column
    df['game_description'] = df['game_description'].str.replace("About This Game", "", case=False).str.lstrip()

    # Cleaning price column
    df['original_price'] = df['original_price'].apply(filter_prices_games)
    # After adjusting the prices, remove all the games where price was changed to Nan ie the games
    # that were demo, no valid price ...
    df = df[df['original_price'].notnull()]
    # removing the outliers
    df = df[df['original_price'] <= 250]
    # plt.boxplot(df['original_price'].values)
    # plt.show()

    # Filter the games that have non english names or invalid names such as emojies
    df['name'] = df['name'].apply(filter_non_valid_game_names)
    df = df[df['name'].notnull()]

    df = trim_features(df)

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv("data/cleaned_steam_games.csv", index=False)
