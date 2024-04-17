import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from cleaning import clean_data
from analyze_origin import analyze_origin_data
from cleaning_playtime import clean_playtime_data
from combine_data import add_playtime_to_steam_data
from recommender import recommend
from predict_popular import predict

sns.set_palette('pastel')
plt.style.use('ggplot')


def main():
    # Load the dataset
    df_all = pd.read_csv("data/steam_games.csv")

    # Data cleaning and generate cleaned data file
    clean_data(df_all)

    df_cleaned = pd.read_csv("data/cleaned_steam_games.csv")

    # Read and clean the second df
    playtime_column_names = ["name", "type", "time", "0"]
    playtime = pd.read_csv("data/steam-200k.csv", names=playtime_column_names)
    clean_playtime_data(playtime)

    playtime_cleaned = pd.read_csv("data/cleaned_playtime.csv")

    combined_data = add_playtime_to_steam_data(df_cleaned, playtime_cleaned)
    combined_data.to_csv("data/combined.csv", index=False)

    while True:
        print("\nWhat do you want to do?")
        print("1. Get a game recommendation.")
        print("2. Train a ML model for game popularity prediction.")
        print("3. Analyze the original data.")
        print("4. Exit.")

        choice = input("\nYour choice: ")

        if choice == "1":
            # Call the game recommendation function
            recommend()
        elif choice == "2":
            # Call the popularity prediction function
            predict()
        elif choice == "3":
            # Perform analysis on the original data
            df_original = df_all.copy()
            analyze_origin_data(df_original)
        elif choice == "4":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please choose a number between 1 and 4.")


if __name__ == "__main__":
    main()
