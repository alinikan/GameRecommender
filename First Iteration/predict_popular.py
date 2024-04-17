import pandas as pd
import re

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

reviews_pattern = re.compile(r"([^,(]+)")
ENABLE_PARAM_TUNING = False


def edit_review_keywords(review):
    review = str(review)
    match = reviews_pattern.search(review)
    if match is not None:
        return match.group(1)
    else:
        return None


def extract_popular(rating_percentage):
    if rating_percentage > 70:
        return 1
    else:
        return 0


def one_hot_encoding(combined_data):
    # One hot encodes languages, details and genres.
    # Learned from: https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/
    one_hot_languages = combined_data['languages'].str.get_dummies(sep=',')
    combined_data = combined_data.join(one_hot_languages)

    one_hot_release_year = pd.get_dummies(combined_data['release_year'])
    combined_data = combined_data.join(one_hot_release_year)

    one_hot_release_month = pd.get_dummies(combined_data['release_month'])
    combined_data = combined_data.join(one_hot_release_month)

    one_hot_game_details = combined_data['game_details'].str.get_dummies(sep=',')
    combined_data = combined_data.join(one_hot_game_details)

    one_hot_game_genre = combined_data['genre'].str.get_dummies(sep=',')
    combined_data = combined_data.join(one_hot_game_genre)

    one_hot_developer = pd.get_dummies(combined_data['developer'])
    combined_data = combined_data.join(one_hot_developer)

    combined_data = combined_data.reset_index()

    combined_data.drop(
        ['index', 'genre', 'release_date', 'release_year', 'release_month', 'developer', 'languages', 'game_details'],
        inplace=True, axis=1)

    return combined_data


def parse_date(date_string):
    for fmt in ('%b %d, %Y', '%b %Y'):
        try:
            return pd.to_datetime(date_string, format=fmt)
        except ValueError:
            continue
    raise ValueError(f'{date_string} is not in the format')


def preprocess(combined_data):
    # remove the columns that are not necessary for predicting game rating
    combined_data = combined_data[
        ['name', 'all_reviews', 'release_date', 'languages', 'game_details', 'genre', 'developer']].copy()

    # Get label class popularity
    combined_data['percentage'] = combined_data['all_reviews'].str.extract(r'(\d+)%').astype(float)
    combined_data['popularity'] = combined_data['percentage'].apply(extract_popular)

    combined_data = combined_data.drop(['percentage'], axis=1)

    # # Remove the games where we don't have the average play time ie the play_time is Nan
    # combined_data = combined_data[combined_data['average_playtime'].notna()]

    # Extract release year column
    combined_data['release_date'] = combined_data['release_date'].apply(parse_date)
    # combined_data['release_date'] = pd.to_datetime(combined_data['release_date'])
    combined_data['release_year'] = combined_data['release_date'].dt.year.astype(str)
    combined_data['release_month'] = combined_data['release_date'].dt.month.astype(str)

    # Rename all development team with appearence of 1 to small_company
    developer_counts = combined_data['developer'].value_counts()
    small_companies = developer_counts[developer_counts == 1].index.tolist()
    combined_data['developer'] = combined_data['developer'].replace(small_companies, 'small_company')

    combined_data = one_hot_encoding(combined_data)

    return combined_data


def param_tunning(X_train, y_train, class_weight):
    # Define the parameter grid for RandomForestClassifier
    # https://scikit-learn.org/stable/modules/grid_search.html
    param_grid_random_forest = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 5, 10, 15, 25, 30],
        'min_samples_split': [1, 2, 4, 13, 17, 25],
        'min_samples_leaf': [2, 5, 10, 15, 20, 25],
        'class_weight': [class_weight],
    }

    param_grid_knn = {
        'n_neighbors': [2, 3, 5, 7, 10, 15, 20, 35, 40, 50, 60]
    }

    param_grid_decision_tree = {
        'max_depth': [2, 4, 8, 10],
        'min_samples_split': [1, 3, 5, 10],
        'min_samples_leaf': [8, 15, 20, 35],
        'class_weight': [class_weight],

    }

    param_grid_svm = {
        'C': [0.01, 0.1, 0.5, 1, 5, 10, 15],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear', 'poly'],
        'class_weight': [class_weight],
    }

    # Create a base model
    random_forest = RandomForestClassifier()
    knn = KNeighborsClassifier()
    # decision_tree = DecisionTreeClassifier()
    svm = SVC()

    # Instantiate the grid search model
    grid_search_random_forest = GridSearchCV(estimator=random_forest, param_grid=param_grid_random_forest, cv=5,
                                             n_jobs=-1, scoring='f1_macro',
                                             verbose=2)
    grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, n_jobs=-1, scoring='f1_macro',
                                   verbose=2)
    # grid_search_decision_tree = GridSearchCV(estimator=decision_tree, param_grid=param_grid_decision_tree, cv=5,
    #                                          n_jobs=-1,scoring='f1_macro',
    #                                          verbose=2)
    grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, n_jobs=-1, scoring='f1_macro',
                                   verbose=2)

    # Fit the grid search to the data
    grid_search_random_forest.fit(X_train, y_train)
    grid_search_knn.fit(X_train, y_train)
    # grid_search_decision_tree.fit(X_train, y_train)
    grid_search_svm.fit(X_train, y_train)

    # Get the best parameters
    best_params = {
        'random_forest': grid_search_random_forest.best_params_,
        'knn': grid_search_knn.best_params_,
        # 'decision_tree': grid_search_decision_tree.best_params_,
        'svm': grid_search_svm.best_params_,
    }

    # Create the VotingClassifier with the best parameters for RandomForestClassifier
    model = VotingClassifier([
        ('knn', KNeighborsClassifier(**best_params['knn'])),
        # ('decision_tree', DecisionTreeClassifier(**best_params['decision_tree'])),
        ('random_forest', RandomForestClassifier(**best_params['random_forest'])),
        ('svm', SVC(**best_params['svm'])),
    ])

    print(best_params['random_forest'])
    print(best_params['knn'])
    # print(best_params['decision_tree'])
    print(best_params['svm'])

    return model


# Auto Feature selection based on importance learned from: https://www.datatechnotes.com/2021/02/seleckbest-feature-selection-example-in-python.html
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


def predict_game_rating(combined_data):
    X = combined_data.drop(['name', 'all_reviews', 'popularity'],
                           axis=1, inplace=False)
    X = X.values
    y = combined_data['popularity'].values
    # print("Label portions: \n", combined_data['popularity'].value_counts(normalize=True))

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=30)
    class_weights = dict(1 / combined_data['popularity'].value_counts(normalize=True))

    # Normalization
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_valid = scaler.transform(X_valid)

    # Transform into data frame to perform auto feature importance selection
    # X_train_df = pd.DataFrame(X_train, columns=features)
    # X_valid_df = pd.DataFrame(X_valid, columns=features)
    # X_train_fs, X_test_fs, fs = select_features(X_train_df, y_train, X_valid_df)
    # unimportant_features = []
    # for i in range(len(fs.scores_)):
    #     if fs.scores_[i] < 0.01:
    #         unimportant_features.append(features[i])
    # X_train_df = X_train_df.drop(columns=unimportant_features)
    # X_valid_df = X_valid_df.drop(columns=unimportant_features)
    # print("Training shape after feature selection:", X_train_df.shape)
    #
    # X_train = X_train_df.values
    # X_valid = X_valid_df.values
    # print(X_train[0])

    if ENABLE_PARAM_TUNING:
        model = param_tunning(X_train, y_train, class_weights)
    else:
        # Best parameters:
        knn = KNeighborsClassifier(5)
        random_forest = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=4, min_samples_leaf=5,
                                               class_weight=class_weights)
        svm = SVC(C=0.1, gamma='scale', kernel='poly', class_weight=class_weights)
        model = VotingClassifier([
            ('knn', knn),
            # ('decision_tree', decision_tree),
            ('random_forest', random_forest),
            ('svm', svm),
        ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)  # Predict the labels of the validation set

    f1 = f1_score(y_valid, y_pred, average='macro')  # Calculate the F1 score

    print("f1-macro", f1)
    print("Train Accuracy:", model.score(X_train, y_train))
    print("Validation Accuracy:", model.score(X_valid, y_valid))


# def predict_pos_reviews(combined_data):
#     X = combined_data.drop(['name', 'all_reviews', 'languages', 'game_details', 'positive_reviews'], axis=1,
#                            inplace=False).values
#     y = combined_data['positive_reviews']
#
#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#     # Define the individual regressors
#     reg1 = RandomForestRegressor()
#
#     # Fit the ensemble to the training data
#     reg1.fit(X_train, y_train)
#
#     # Make predictions on the test data
#     y_pred = reg1.predict(X_test)

def predict():
    pd.set_option('display.max_columns', None)

    combined_data = pd.read_csv("data/combined.csv")

    # Data preprocessing
    combined_data = preprocess(combined_data)

    predict_game_rating(combined_data)
