import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def extract_title(full_name):
    names = full_name.split(' ')
    for name in names:
        if name[-1] == '.':
            return name[:-1]
    return ''


def engineer_features(df):
    '''Process features '''
    # Cabin: Generate Deck, Room from Cabin
    df['Deck'] = df['Cabin'].str.slice(0,1)
    df['Deck'].fillna('U', inplace=True)
    df['Room'] = df['Cabin'].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
    df['Room'].fillna(df["Room"].mean(), inplace=True)
    
    # Embarked: Fill in missing values with the most popular port
    df['Embarked'].fillna('S', inplace=True)
    
    # Name: Generate Title
    df['Title'] = df['Name'].apply(extract_title)
    popular_titles = ['Mrs', 'Miss', 'Master', 'Mr']
    df['Title'] = np.where(~df['Title'].isin(popular_titles), 'Other', df['Title'])
    
    return df


def main(training_data, test_data):
    # Combine training and test datasets to have more training data with valid age values
    train_df = pd.read_csv(training_data)
    test_df = pd.read_csv(test_data)
    combined_df = pd.concat([train_df, test_df])

    selected_features = ['PassengerId', 'Age', 'Deck', 'Room', 'Embarked', 'Fare', 'Title',
                         'Parch', 'Pclass', 'Sex', 'SibSp', 'Survived']
    # One-hot encode the engineered features
    processed_combined_df = pd.get_dummies(engineer_features(combined_df)[selected_features])
    print(processed_combined_df.columns)

    # Filter out unwanted features
    features = list(processed_combined_df.columns)
    for feature in ['PassengerId', 'Age', 'Survived']:
        features.remove(feature)

    # Fill in missing value with the average Fare of class 3 
    avg_class3_fare = processed_combined_df[processed_combined_df['Pclass'] == 3]['Fare'].mean()
    processed_combined_df['Fare'].fillna(avg_class3_fare, inplace=True)
    null_age_df = processed_combined_df[processed_combined_df['Age'].isnull()]
    not_null_age_df = processed_combined_df[processed_combined_df.Age.notnull()]
    print("Number of passengers with age data: %s" %not_null_age_df.shape[0])
    print("Number of passengers with no age data: %s" %null_age_df.shape[0])

    # Fit Random Forest model to predict missing age
    age_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    age_rf.fit(not_null_age_df[features], not_null_age_df['Age'])
    null_age_df.loc[:, 'Age'] = age_rf.predict(null_age_df[features])

    return pd.concat([not_null_age_df, null_age_df], sort=False), features