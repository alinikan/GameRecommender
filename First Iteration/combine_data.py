import pandas as pd
from fuzzywuzzy import fuzz, process

MATCH_THRESHOLD = 70


def find_potential_matches(steam_data, playtime_data, threshold):
    matches = {}
    for name in steam_data['name']:
        best_match, score, _ = process.extractOne(name, playtime_data['name'], scorer=fuzz.ratio)
        if score >= threshold:
            matches[name] = best_match
    return matches


def add_playtime_to_steam_data(steam_data, playtime_data):
    potential_matches = find_potential_matches(steam_data, playtime_data, MATCH_THRESHOLD)
    steam_data['average_playtime'] = steam_data['name'].map(potential_matches).map(playtime_data.set_index('name')['time'])
    # steam_data = steam_data.dropna(subset=['average_playtime'])

    return steam_data
