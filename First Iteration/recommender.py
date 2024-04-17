import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from guess import find_game
import warnings

# Suppress specific warning from sklearn (this is just a warning and does not impact the functionality of the code)
warnings.filterwarnings('ignore', '.*X does not have valid feature names.*')


# Function to preprocess features
def preprocess_features(df):
    # Select only the columns we want to preprocess
    cols_to_preprocess = ['popular_tags', 'game_details', 'genre']
    df_subset = df[cols_to_preprocess]

    # Fill NA values with 'unknown', convert to lower case, strip whitespace, and split on comma
    df_subset = df_subset.fillna('unknown').applymap(lambda x: x.lower().strip().split(','))

    # Replace the original columns in df with the preprocessed ones
    df[cols_to_preprocess] = df_subset

    return df


def google_search_link(game_name):
    base_url = "https://www.google.com/search?q="
    search_query = game_name.replace(" ", "+")  # replace spaces with '+'
    return base_url + search_query


# Function to get the index of a game given its name
def get_index_from_name(df, name):
    return df[df['name'] == name].index[0]


# Function to get the name of a game given its index
def get_name_from_index(df, index):
    return df.loc[index, 'name']


# Function to recommend games based on a given game
def recommend_games_with_explanation(df, df_encoded, nn_model, game_name, num_recommendations=5):
    # Get the index of the input game
    game_index = get_index_from_name(df, game_name)

    # Use the NearestNeighbors model to find the nearest neighbors of the input game
    # Request num_recommendations + 1 neighbours since the closest one will be the game itself, which should be ignored
    distances, indices = nn_model.kneighbors(df_encoded.iloc[game_index, :].values.reshape(1, -1),
                                             n_neighbors=num_recommendations + 1)

    # Convert the flattened indices array (excluding the first element) to a pandas Series
    indices_series = pd.Series(indices.flatten()[1:])

    # Apply the get_name_from_index function to each element in the Series, to get the game name for each index
    # Then, convert the resulting pandas Series back to a list
    recommended_games = indices_series.apply(lambda i: get_name_from_index(df, i)).tolist()

    # Get the features (tags, genres, and game details) of the input game
    input_game_features = set(df_encoded.columns[df_encoded.loc[game_index] == 1])

    # Initialize the list of recommendations with explanations
    recommendations_with_explanations = []

    for recommended_game in recommended_games:
        # Get the index of the recommended game
        recommended_game_index = get_index_from_name(df, recommended_game)
        # Get the features of the recommended game
        recommended_game_features = set(df_encoded.columns[df_encoded.loc[recommended_game_index] == 1])
        # Find the common features between the input game and the recommended game
        common_features = input_game_features.intersection(recommended_game_features)
        # Create an explanation for the recommendation
        explanation = (
            f"{recommended_game} is recommended because it has the following common features with {game_name}:\n"
            f"{', '.join(common_features)}.\n"
            f"Learn more about it here: {google_search_link(recommended_game)}")
        # Append the recommendation and explanation to the list
        recommendations_with_explanations.append((recommended_game, explanation))

    # Return the list of recommendations with explanations
    return recommendations_with_explanations


def recommend():
    # Load the csv file into a DataFrame
    df = pd.read_csv('data/cleaned_steam_games.csv')

    # Preprocess the 'name' column and the feature columns
    df['name'] = df['name'].str.lower().str.strip()
    df = preprocess_features(df)

    # Initialize the MultiLabelBinarizer for each feature column
    # It transforms collections of labels into a binary format that can be used for machine learning
    mlb_tags = MultiLabelBinarizer()
    mlb_details = MultiLabelBinarizer()
    mlb_genre = MultiLabelBinarizer()

    # Transform the 'popular_tags', 'game_details', and 'genre' columns into binary vectors
    df_encoded_tags = pd.DataFrame(mlb_tags.fit_transform(df['popular_tags']), columns=mlb_tags.classes_,
                                   index=df.index)
    df_encoded_details = pd.DataFrame(mlb_details.fit_transform(df['game_details']), columns=mlb_details.classes_,
                                      index=df.index)
    df_encoded_genre = pd.DataFrame(mlb_genre.fit_transform(df['genre']), columns=mlb_genre.classes_, index=df.index)

    # Concatenate the three DataFrames along the columns axis
    df_encoded = pd.concat([df_encoded_tags, df_encoded_details, df_encoded_genre], axis=1)

    # Fit the NearestNeighbors model to our encoded DataFrame
    nn_model = NearestNeighbors(metric='euclidean')
    nn_model.fit(df_encoded)

    # Call the find_game function and print the result
    game = find_game(df)
    if game:
        print(f"\n--- Found the game: {game.upper()} ---\n")
        recommendations = recommend_games_with_explanation(df, df_encoded, nn_model, game, num_recommendations=5)
        for game, explanation in recommendations:
            print(f"\n--- Recommended Game: {game.upper()} ---")
            print(f"\n{explanation}\n")
            print(f"{'-' * 50}\n")  # print a separator line
    else:
        # If the game is not found, print an error message
        print("Sorry, I couldn't find the game you're looking for.")
