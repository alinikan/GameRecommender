import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import stats


def clean_playtime_data(df):
    # Drop rows of purchase record to keep only play time
    df_origin = df[df['type'] == 'play']

    # Drop the last column as it does not contain any data rather than 0
    # Also drop the type column as it does not contain any useful info anymore after filtering
    df_origin = df_origin.drop(['0', 'type'], axis=1)

    # Calculate Z-score to remove outliers that are 4 std away from the mean of the data
    z_origin = stats.zscore(df_origin['time'])
    outlier_rm_data = df_origin[np.abs(z_origin) <= 4]

    # Group play time data on video games, and take the average of player's play time.
    df_grouped = outlier_rm_data.groupby('name')
    group_sizes = outlier_rm_data.groupby('name').size()
    df_averaged = df_grouped.mean()
    df_averaged = df_averaged[group_sizes >= 10]

    # Save the cleaned DataFrame to a new CSV file
    df_averaged.to_csv("data/cleaned_playtime.csv")

    # Plot the average play time of games
    plt.clf()
    df_averaged['time'].plot(kind='hist', bins=200, logy=True)
    plt.title('Average Play time vs Game Amount (Sample > 10)')
    plt.xlabel('Average Play Time')
    plt.ylabel('Number of Games')
    plt.savefig('outputs/AvgPlaytimeOutlier.png')

    # Focus on the first 30 hrs
    plt.clf()
    zoomed_df = df_averaged[df_averaged['time'] < 30]
    counts, bins, patches = plt.hist(zoomed_df['time'], bins=80, log=True, alpha=0.5)
    plt.plot(bins[:-1], counts, color='red')
    plt.xlabel('Average Play Time')
    plt.ylabel('Number of Games')
    plt.title('Zoomed Plot in 30 Hours')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('outputs/ZoomedAvgPlaytimeOutlier.png')
