import pandas as pd
from fuzzywuzzy import process


def find_game(df, attempts=5):
    # Ask the user for the game's name
    # Convert the input to lower case and remove any leading or trailing spaces
    game_name = input("Please enter a game's name: ").lower().strip()

    # Convert the 'name' column of the DataFrame to a list
    names = df['name'].tolist()

    # Repeat the matching process for a specified number of attempts
    for i in range(attempts):
        # Use fuzzy string matching to find the closest match to the user's input
        # process.extractOne returns the best match and the match's score
        match, score = process.extractOne(game_name, names)

        # Ask the user to confirm whether the match is correct
        user_confirmation = input(f"Did you mean {match}? (yes/no): ")

        # If the user confirms the match, return the match
        if user_confirmation.lower() == "yes":
            return match
        else:
            # If the user does not confirm the match, remove it from the list
            # This allows the next best match to be found in the next iteration
            names.remove(match)

    # If no match is confirmed after the specified number of attempts, return None
    return None


# Load the csv file into a DataFrame
df = pd.read_csv('data/cleaned_steam_games.csv')

# Preprocess the 'name' column
# Convert all names to lower case and remove any leading or trailing spaces
df['name'] = df['name'].str.lower().str.strip()
