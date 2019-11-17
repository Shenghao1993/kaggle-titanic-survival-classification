import numpy as np
import pandas as pd
import re
import time
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix


def extract_title(full_name):
    title_search = re.search(' ([A-Za-z]+)\.', full_name)
    if title_search:
        title = title_search.group(1)
        title.replace('Ms', 'Miss')
        # Normalise any title in French
        title.replace('Mlle', 'Miss')
        title.replace('Mme', 'Mrs')
        return title
    return ''


def normalize_room_no(deck, room_no, max_room_no_dict):
    return room_no / max_room_no_dict.get(deck)


def engineer_features(df):
    '''Process features '''
    # Map Sex
    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Fill in missing Age value
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null_count = df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df['CategoricalAge'] = df['Age']
    df['CategoricalAge'][np.isnan(df['CategoricalAge'])] = age_null_random_list

    # Map Age
    df.loc[ df['CategoricalAge'] <= 16, 'CategoricalAge'] = 0
    df.loc[(df['CategoricalAge'] > 16) & (df['CategoricalAge'] <= 32), 'CategoricalAge'] = 1
    df.loc[(df['CategoricalAge'] > 32) & (df['CategoricalAge'] <= 48), 'CategoricalAge'] = 2
    df.loc[(df['CategoricalAge'] > 48) & (df['CategoricalAge'] <= 64), 'CategoricalAge'] = 3
    df.loc[ df['CategoricalAge'] > 64, 'CategoricalAge'] = 4
    df['CategoricalAge'] = df['CategoricalAge'].astype(int)

    # Create FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    min_max_scaler = preprocessing.MinMaxScaler()
    df['NormalizedFamilySize'] = min_max_scaler.fit_transform(df[['FamilySize']].values)
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Name: Generate Title
    df['Title'] = df['Name'].apply(extract_title)
    popular_titles = ['Mrs', 'Miss', 'Master', 'Mr']
    # Group all non-common titles into one single group "Rare"
    df['Title'] = np.where(~df['Title'].isin(popular_titles), 'Rare', df['Title'])
    df['NameLength'] = df['Name'].apply(len)
    df['NormalizedNameLength'] = min_max_scaler.fit_transform(df[['NameLength']].values)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

    # Mapping Fare
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    # Cabin: Generate Deck, Room from Cabin
    df['HasCabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    df['Deck'] = df['Cabin'].str.slice(0,1)
    df['Deck'].fillna('U', inplace=True)
    df['Room'] = df['Cabin'].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
    max_room_no_df = df[["Deck", "Room"]].groupby(['Deck'], as_index=False).max()
    max_room_no_dict = max_room_no_df.set_index('Deck').to_dict().get('Room')
    print(max_room_no_dict)
    df['NormalizedRoomNo'] = df.apply(lambda x: normalize_room_no(x.Deck, x.Room,
                                                                  max_room_no_dict), axis=1)
    df['NormalizedRoomNo'].fillna(df['NormalizedRoomNo'].mean(), inplace=True)
    
    # Embarked: Fill in missing values with the most popular port
    df['Embarked'].fillna('S', inplace=True)
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    return df


def preprocess(train_data_path, test_data_path):
    # Load training and test datasets
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    processed_dfs = []
    drop_cols = ['PassengerId', 'NameLength', 'SibSp', 'Parch', 'Ticket',
                 'FamilySize', 'Cabin', 'Deck', 'Room']
    for df in [train_df, test_df]:
        # Fill in missing Fare value with the median of training set.
        df['Fare'] = df['Fare'].fillna(train_df['Fare'].median())

        # One-hot encode the categorical features
        encoded_df = pd.get_dummies(engineer_features(df),
                                    columns=['Pclass', 'CategoricalAge', 'Fare', 'Embarked', 'Title'])
        print("One-hot encoded columns: ", encoded_df.columns)
        encoded_df = encoded_df.drop(drop_cols, axis = 1)
        processed_dfs.append(encoded_df)
    return processed_dfs


def select_params(model, param_grid, X_train, y_train, nfolds, grid_search=True):
    # scores = ['accuracy', 'precision', 'recall']
    scores = ['accuracy']
    start_time = time.time()
    best_score = {}
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        if grid_search:
            clf = GridSearchCV(model, param_grid, cv=nfolds, scoring='%s' % score)
        else:
            clf = RandomizedSearchCV(model, param_grid, cv=nfolds, scoring='%s' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        best_score[score] = max(means)
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print("Elapsed time: %s seconds" % round(time.time() - start_time, 4))
        print()
    return clf, best_score


def evaluate(model, features, y_true):
    '''Evaluate performance of classification
    '''
    y_pred = model.predict(features)
    conf = confusion_matrix(y_true, y_pred)
    (tn, fp, fn, tp) = conf.ravel() 
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    print(conf)
    print("True positive: ", tp)
    print("True negative: ", tn)
    print("False positive: ", fp)
    print("False negative: ", fn)
    print("Accuracy = (tp + tn) / (p + n), Accuracy = ", round(accuracy, 6))
    print("Precision = tp / (tp + fp), Precision = ", round(precision, 6))
    print("Recall = tp / (tp + fn), Recall = ", round(recall, 6))
    print("ROC = ", round(roc_auc, 6))