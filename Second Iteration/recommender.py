import warnings

warnings.filterwarnings("ignore",
                        message="Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from fuzzywuzzy import process

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except IOError:
    print("Downloading the 'en_core_web_sm' model...")
    from spacy.cli import download

    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load the dataset
df = pd.read_csv('data/final.csv')


def preprocess_text_spacy(text):
    text = str(text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])


df['game_details_keywords'] = df['game_details'].apply(lambda x: preprocess_text_spacy(str(x)))
df['genre_keywords'] = df['genre'].apply(lambda x: preprocess_text_spacy(str(x)))
df['popular_tags_keywords'] = df['popular_tags'].apply(lambda x: preprocess_text_spacy(str(x)))
df['combined_keywords'] = df[['game_details_keywords', 'genre_keywords', 'popular_tags_keywords']].agg(' '.join, axis=1)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_keywords'])


def calculate_similarity_score(user_input, recommended_games):
    user_input_vectorized = tfidf_vectorizer.transform([user_input])
    scores = []
    for _, game in recommended_games.iterrows():
        game_vectorized = tfidf_vectorizer.transform([game['combined_keywords']])
        score = cosine_similarity(user_input_vectorized, game_vectorized)
        scores.append(score[0][0])
    average_score = sum(scores) / len(scores) if scores else 0
    return average_score


def recommend_games(user_input, top_n=5):
    user_input_vectorized = tfidf_vectorizer.transform([user_input])
    cos_similarity = cosine_similarity(user_input_vectorized, tfidf_matrix)
    top_indices = cos_similarity[0].argsort()[-top_n:][::-1]
    recommended_games = df.iloc[top_indices][['name', 'genre', 'popular_tags', 'combined_keywords']]
    similarity_score = calculate_similarity_score(user_input, recommended_games)
    return recommended_games, similarity_score


def interactive_recommendation():
    print("\n\n" + "=" * 50)
    print("Welcome to the game recommendation system!")
    print("=" * 50 + "\n")
    print("You can choose a recommendation basis from the following options:\n")
    print("'Genre' - Get recommendations based on game genres. Example inputs: 'Action', 'Adventure', 'Strategy'")
    print(
        "'Details' - Focus on specific game details or features. Example inputs: 'Multiplayer', 'Single-player', 'Co-op'")
    print(
        "'Tags' - Use popular tags associated with games for recommendations. Example inputs: 'Open World', 'RPG', 'Sci-fi'\n")

    choices = ['genre', 'details', 'tags']
    user_choice = input("Please enter your choice (genre/details/tags): ").lower()
    print("\n")

    closest_choice, score = process.extractOne(user_choice, choices)
    if score < 90:
        confirmation = input(f"Did you mean '{closest_choice}'? (yes/no): ").lower()
        if confirmation != 'yes':
            print("\nNo problem, let's try again. Make sure to choose from 'genre', 'details', or 'tags'.\n")
            return
    else:
        print(f"You've selected '{closest_choice}'.\n")

    print(f"Enter your preference for {closest_choice}:")
    user_input = input()
    user_input = preprocess_text_spacy(user_input)
    recommended_games, similarity_score = recommend_games(user_input)

    print("\n\n" + "-" * 50)
    print("Recommended games based on your input:\n")
    for index, row in recommended_games.iterrows():
        print(f"- {row['name']} (Genre: {row['genre']}, Tags: {row['popular_tags']})")
    print("-" * 50)
    print(f"\nAverage similarity score for these recommendations: {similarity_score:.2f}\n\n")


interactive_recommendation()
